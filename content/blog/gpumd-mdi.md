---
title: "GPUMD-MDI: A GPU Molecular-Dynamics Engine Behind the MolSSI Driver Interface"
date: 2026-10-15
description: "Exposing GPUMD as an MDI ENGINE so external codes such as VASP can supply energies and forces while GPUMD handles GPU-accelerated integration — the foundation for external-force and QM/MM simulations."
tags: ["MD", "GPU", "MDI", "QM/MM", "NEP"]
math: true
---

### Why couple a GPU MD engine to other codes?

Modern simulation workflows rarely live inside a single program. You might want a machine-learned potential for the bulk of a system and a quantum-chemistry (QM) solver for a small reactive region; or an enhanced-sampling library to steer a rare event; or a path-integral or transition-state code wrapped around the dynamics. In every case there is an **MD engine** — the code that moves the atoms — and a **driver** — the code that decides *where* they move or *what forces* act on them.

The naive way to connect them is a bespoke interface per pair of codes. With $M$ engines and $N$ drivers that is an $M \times N$ integration problem, each solution with its own file formats, units, and time-step conventions. A shared protocol collapses that to $M + N$.

### What MDI is

The [MolSSI Driver Interface](https://molssi-mdi.github.io/MDI_Library/) (MDI) is exactly such a protocol. It defines

- two **roles** — an `ENGINE` that exposes data and does work, and a `DRIVER` that orchestrates;
- a small **command vocabulary** for exchanging atom counts, coordinates, forces, energies, stresses, and cells;
- a pluggable **transport** — TCP sockets, Unix named pipes, MPI, or shared memory.

A DRIVER connects to an ENGINE, sends a command such as `<COORDS` ("engine, send me the coordinates"), and reads the reply. Data flowing the other way is marked with `>` (`>FORCES` means "engine, here are forces for you"). The protocol is language-agnostic, and the reference library ships C, C++, Fortran, and Python bindings.

In this article the ENGINE is **GPUMD**: a GPU-native MD code written in CUDA with first-class support for the neuroevolution potential (NEP) family of machine-learned potentials. Making it an MDI ENGINE buys three things:

1. **GPU-accelerated integration for free.** An external code that can produce forces does not have to reimplement a velocity-Verlet loop, a thermostat, or file I/O.
2. **A route to QM/MM.** A QM code computes forces for a small reactive region while GPUMD integrates the whole system.
3. **Reuse of `run.in`.** GPUMD's `dump_xyz`, `dump_thermo`, and `compute_*` keywords keep working even though the forces come from somewhere else.

## Architecture

The repository keeps the interface *outside* the GPUMD tree. The build copies a handful of files into a GPUMD checkout and links a separate executable, `gpumd-mdi`. The DRIVER and the ENGINE never touch each other's memory directly — every exchange travels through the MDI library, which can carry the conversation over TCP, Unix named pipes, MPI, or shared memory. The ENGINE side is built from four files that are copied into `GPUMD/src/`:

```text
src/main_mdi/
├─ main.cu          entry point: parses -in, detects --mdi, hands off to MDI
├─ mdi_stub.cu      the MDI command loop and command registration
├─ run.cu / run.cuh a copy of GPUMD's Run with mdi_* single-step helpers
└─ mdi_fallback.h   local declarations (real symbols come from libmdi)
src/makefile_mdi    build rules for the gpumd-mdi executable
```

Keeping the sources in a dedicated `main_mdi/` directory is deliberate: the GPUMD core stays untouched, and the interface can be updated independently as GPUMD evolves (see [GPUMD issue #1786](https://github.com/brucefan1983/GPUMD/issues/1786) for the motivation).

## The MDI command set

GPUMD registers a `@DEFAULT` node and the following commands:

| Command    | Meaning                                            |
|-----------|----------------------------------------------------|
| `<NATOMS`  | Returns the number of atoms                        |
| `>COORDS`  | Receives atomic coordinates from the driver        |
| `<COORDS`  | Sends atomic coordinates to the driver             |
| `<FORCES`  | Computes and sends forces to the driver            |
| `>FORCES`  | Receives forces from the driver (external coupling)|
| `<ENERGY`  | Computes and sends potential energy to the driver  |
| `>ENERGY`  | Receives energy from the driver                    |
| `>STRESS`  | Receives the stress tensor from the driver         |
| `EXIT`     | Exits the MDI communication loop                   |

A minimal engine-side initialization looks like this:

```c
MDI_Init(&argc, &argv);
run->mdi_initialize_for_mdi();

MDI_Register_node("@DEFAULT");
MDI_Register_command("@DEFAULT", "<NATOMS");
MDI_Register_command("@DEFAULT", ">COORDS");
MDI_Register_command("@DEFAULT", "<COORDS");
MDI_Register_command("@DEFAULT", ">FORCES");
MDI_Register_command("@DEFAULT", "<FORCES");
MDI_Register_command("@DEFAULT", "<ENERGY");
MDI_Register_command("@DEFAULT", ">ENERGY");
MDI_Register_command("@DEFAULT", ">STRESS");
MDI_Register_command("@DEFAULT", "EXIT");

MDI_Accept_communicator(&mdi_comm);   /* blocks until a DRIVER connects */

while (!exit_flag) {
  MDI_Recv_command(command, mdi_comm);
  /* dispatch on `command` and exchange data */
}
```

The engine is launched with `--mdi` followed by a single, quoted options string:

```bash
gpumd-mdi --mdi \
  "-role ENGINE -name gpumd -method TCP -hostname localhost -port 8021" \
  -in run.in
```

One convenient detail: the MDI library itself recognises `--mdi`, tokenizes the options string, configures the role/transport, and then strips those two arguments before GPUMD parses its own `-in` flag. So the same binary can run either as a normal GPUMD (`gpumd-mdi -in run.in`) or as an MDI ENGINE.

## Wiring MDI into GPUMD's run loop

This was the interesting engineering problem. GPUMD's `Run` class is built around **whole runs**: it reads `run.in`, and the `run N` keyword performs `N` steps immediately. MDI, by contrast, is *step-driven* — the DRIVER decides when and how many times to integrate.

The fix has three parts.

### Keeping the ensemble alive

`Run` normally requires every `ensemble` to be followed by a `run`:

```text
The last ensemble is not followed by a run.
```

In MDI mode that check is skipped, and the `run` keyword no longer executes — it only records the requested length (used for one-time initialization and temperature ramps). The ensemble and integrator therefore stay live, ready to be advanced one step at a time.

### Single-step integration

`mdi_step_one()` mirrors exactly one iteration of GPUMD's normal MD loop — the same sequence of `correct_velocity → integrate.compute1 → measure hooks → (force) → integrate.compute2 → measure.end_of_step` — so that `dump_*` and `compute_*` actions behave identically to a conventional run. Two details matter for correctness:

- **Monotonic step counter.** The integrator, thermostat chains, and the velocity-correction interval all expect a step index that increases. MDI uses its own counter rather than resetting to zero on every call.
- **Bounded ramp denominator.** The temperature ramp of a temperature-dependent NEP potential is normalized by the requested `run` length. Once the DRIVER goes past that length, the denominator tracks the step counter so the ratio saturates at 1 instead of growing without bound.

Initialization and teardown are likewise split out into `mdi_initialize_for_mdi()` and `mdi_finalize_for_mdi()`, which reproduce the setup/cleanup that `perform_a_run()` does around the loop.

### External forces, energies, and stresses

When the DRIVER sends `>FORCES`, the engine copies the array into `force_per_atom` and sets a pending flag. On the next integration step the internal (NEP) force computation is skipped and the external forces are used instead:

```c
void Run::mdi_set_forces(const double* forces) {
  atom.force_per_atom.copy_from_host(forces, (size_t)(N * 3));
  external_forces_pending = true;
}
```

`>ENERGY` and `>STRESS` are handled the same way, so GPUMD's thermodynamic output stays consistent even though the physics came from outside. The external energy is distributed across the per-atom potential array, and the stress tensor is converted to GPUMD's virial convention ($\boldsymbol{\sigma} = -\mathbf{W}/V$) before being written into `virial_per_atom`. That means a `dump_thermo 1` line printed during an MDI run reports the *external* energy and stress, not the residual NEP values.

## A QM/MM-style workflow: VASP as DRIVER

The bundled example couples GPUMD to **VASP** for a small Cu system. VASP plays the DRIVER: at every MD step it computes the QM energy, forces, and stress, and hands them to GPUMD, which performs the actual time integration.

```text
DRIVER (VASP)                              ENGINE (GPUMD)
   │  <NATOMS ────────────────────────────▶  │  return natoms
   │                                         │
   │  <COORDS ────────────────────────────▶  │  send positions
   │  ◀──────────────────────── r[3N]        │
   │                                         │
   │  [write POSCAR → run VASP → parse XML]  │
   │                                         │
   │  >ENERGY ──── E_total ────────────────▶ │  store external energy
   │  >STRESS ──── σ(3×3) ─────────────────▶ │  store external stress
   │  >FORCES ──── F[3N] ──────────────────▶ │  store forces → integrate one step
   │                                         │
   │  EXIT ────────────────────────────────▶ │  finalize and stop
```

The DRIVER lives in `tools/gpumd-mdi/vasp_mdi_driver.py`. Each step it writes the current coordinates into a `POSCAR`, runs VASP with `IBRION = -1` (so the external driver owns the ionic motion), parses `energy`, `varray[@name='forces']`, and `varray[@name='stress']` out of `vasprun.xml`, and sends them back. If VASP fails or the XML is incomplete, the driver sends zero forces and continues rather than crashing the coupled run.

A small `run.in` is all the engine needs:

```text
potential   /path/to/nep89_20250409.txt
velocity    1

ensemble nve
time_step  0.5
dump_xyz 1 dump.xyz force precision double
dump_thermo 1

run 5
```

Note that the `potential` line is still required — GPUMD loads the NEP potential at startup even though the forces for these steps come from the DRIVER. The `run 5` line sets the requested run length used for initialization and ramps; the number of steps actually taken is controlled by the DRIVER.

The whole thing is wrapped by a launcher that starts the ENGINE in the background, waits for it to listen, then runs the DRIVER:

```bash
cd examples/gpumd_mdi
bash ../../tools/gpumd-mdi/run_mdi_vasp_gpumd.sh \
  --gpumd-bin /path/to/GPUMD/src/gpumd-mdi \
  --run-in run.in \
  --vasp-cmd "vasp_std" \
  --poscar POSCAR \
  --steps 3
```

Because GPUMD keeps its normal `run.in` pipeline, the resulting `dump.xyz` and `dump_thermo` reflect the QM-driven trajectory, and the DRIVER's own log is preserved under `.gpumd_logs/`.

## A VASP-free smoke test

VASP is licensed and often unavailable on a test machine, so the repository also ships a tiny DRIVER that needs no QM code: `tools/gpumd-mdi/mdi_test_driver.py`. It computes simple Lennard-Jones pair forces in Python and pushes them through the same `<NATOMS`, `<COORDS`, `>ENERGY`, `>STRESS`, `>FORCES`, `EXIT` sequence. If this works, the MDI plumbing works.

```bash
# terminal 1 — GPUMD as MDI ENGINE (single GPU for small cells)
CUDA_VISIBLE_DEVICES=0 /path/to/GPUMD/src/gpumd-mdi --mdi \
  "-role ENGINE -name gpumd -method TCP -hostname localhost -port 8021" -in run.in

# terminal 2 — Lennard-Jones DRIVER
python3 tools/gpumd-mdi/mdi_test_driver.py --port 8021 --steps 5
```

If everything is connected, the ENGINE prints the command loop, the DRIVER reports energies and forces per step, and the dumped trajectory shows the atoms moving.

## Building

`build.sh` copies the interface into a GPUMD checkout and compiles it there:

```bash
git clone https://github.com/JaafarMehrez/GPUMD-MDI.git
cd GPUMD-MDI

./build.sh /path/to/GPUMD USE_MDI=1 MDI_LIB=1 \
  MDI_LIB_PATH=/path/to/MDI_Library/install/lib64 \
  MDI_INC_PATH=/path/to/MDI_Library/install/include
```

The resulting executable lands in `/path/to/GPUMD/src/gpumd-mdi`. `makefile_mdi` has two independent switches: `USE_MDI=1` enables the MDI code paths, while `MDI_LIB=1` links the real MDI library (`-lmdi`) using `MDI_INC_PATH` / `MDI_LIB_PATH`. To actually *speak* MDI you need both.

When `MDI_LIB_PATH` is provided, the build bakes an `rpath` into the executable so a shared `libmdi` is found at runtime without `LD_LIBRARY_PATH`:

```make
LDFLAGS += -Xlinker -rpath -Xlinker $(MDI_LIB_PATH)
```

(`nvcc` rejects `-Wl,-rpath,...` directly, hence the host-linker forwarding.) The provided `Dockerfile` instead builds MDI statically (`-DBUILD_SHARED_LIBS=OFF`) so the binary is fully self-contained:

```bash
docker build -t gpumd-mdi .
docker run --rm -it --gpus all gpumd-mdi /opt/GPUMD-MDI/build.sh /opt/GPUMD
```

Finally, a GitHub Actions workflow rebuilds `gpumd-mdi` against the current GPUMD `master` on a weekly schedule. That is the safety net that catches upstream API drift — the interface is compiled and linked on a CPU runner (no GPU needed to build), so a GPUMD refactor that breaks the interface surfaces automatically instead of silently.

## Design choices

### The interface lives outside GPUMD

Keeping `main_mdi/` and `makefile_mdi` in a separate repository means the GPUMD core is never patched. Pulling a new GPUMD `master` and re-running `build.sh` is the entire update path.

### External forces replace the internal ones

The simplest and most predictable coupling is to let the DRIVER own the potential energy surface for a step: when `>FORCES` arrives, GPUMD integrates with those forces and skips its own force evaluation. The same hook is the natural place to build true QM/MM embedding — for example, computing NEP forces for an MM subsystem in the engine and adding them to the DRIVER's QM forces on a reactive subsystem:

$$U = U_{\text{QM}} + U_{\text{MM}} + U_{\text{QM/MM}}, \qquad \mathbf{F}_i = -\nabla_{\mathbf{r}_i} U.$$

### Output consistency

Rather than letting `dump_thermo` report leftover NEP values, the engine folds the DRIVER's `>ENERGY` and `>STRESS` into the per-atom potential and virial arrays. The trajectory and thermodynamic logs therefore describe the coupled system, not the internal potential.

### Single-step bookkeeping

Because MDI step counts are chosen by the DRIVER, the engine keeps its own monotonic step counter and only uses the `run` length for the one-time setup and the temperature-ramp normalization. This keeps thermostats, velocity correction, and NEP temperature ramps well-behaved across an arbitrarily long DRIVER loop.

## Notes and troubleshooting

- **`libmdi` at runtime.** If you linked a *shared* MDI library and skipped the `rpath` (e.g. an older build), export `LD_LIBRARY_PATH=/path/to/MDI_Library/install/lib64:$LD_LIBRARY_PATH` before launching the ENGINE. The symptom is `libmdi.so.1: cannot open shared object file`.
- **Small cells and multiple GPUs.** GPUMD auto-detects all visible GPUs. With more than one GPU it uses the multi-GPU NEP path, which requires each GPU's domain to span at least **5× the NEP cutoff**. A tiny cell (like the 2-atom Cu dimer in a 10 Å box) then aborts with `The longest direction has less than 5 times of the NEP cutoff per GPU.` Restrict the ENGINE to one GPU (`CUDA_VISIBLE_DEVICES=0 …`) or enlarge the cell. When the ENGINE dies the DRIVER reports `MDI_Recv failed` — always fix the ENGINE side first.
- **`POSCAR` is a live file.** The VASP DRIVER reads `POSCAR` for the lattice and atom types and *rewrites* it every step with the current coordinates, so running the example modifies the tracked file. Restore it with `git checkout examples/gpumd_mdi/POSCAR`.
- **The potential still loads.** Even in external-force mode GPUMD initializes the potential named in `run.in`; point it at a real NEP file (for example `<GPUMD>/potentials/nep/nep89_20250409/nep89_20250409.txt`).

## Extending to other DRIVER codes

Nothing in the interface is VASP-specific. Any code that can act as an MDI DRIVER — another QM package, an enhanced-sampling library, a machine-learned force field in Python — can drive GPUMD by following the same loop: request `<COORDS`, compute something, send back `>ENERGY` / `>STRESS` / `>FORCES`, and finish with `EXIT`. The `mdi_test_driver.py` script is intentionally small enough to serve as a starting template.

## Availability

GPUMD-MDI is released under the **GPL-3.0-or-later** license (matching GPUMD) and maintained at [github.com/JaafarMehrez/GPUMD-MDI](https://github.com/JaafarMehrez/GPUMD-MDI). The interface itself is derived from GPUMD; the MDI integration was contributed by Jaafar Mehrez (Shanghai Jiao Tong University / HPQC Labs).

If you use it in your research, please cite GPUMD, the relevant DRIVER code (e.g. VASP), the MDI library, and this interface:

```bibtex
@software{mehrez2026gpumdmdi,
  title = {GPUMD-MDI: MDI (MolSSI Driver Interface) integration for GPUMD},
  author = {Mehrez, Jaafar},
  year = {2026},
  version = {0.1.0},
  license = {GPL-3.0-or-later},
  url = {https://github.com/JaafarMehrez/GPUMD-MDI}
}
```
