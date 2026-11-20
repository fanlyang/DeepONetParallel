# torus_mg_jacobi — build, debug, run report

Status: built and run in the exact image
`fanyoung/dealii-deeponet:latest`. All requested correctness checks passed.
Every timing below is **smoke-test data**, because the image contains ARM
aarch64 binaries and this host is x86_64, so all runs execute under QEMU user
emulation. See `logs/image_platform_note.txt`.

## Where this project lives

The project root `D:\MS\Munster\ms` currently also contains a different
program (`torus_patch_mpi`, a single-level matrix-vector verifier) and a
`deepseek/` copy of an earlier torus solve. Those files were **not modified**.
The pasted `torus_mg_jacobi` solver is isolated in this directory:

```
D:\MS\Munster\ms\torus_mg_jacobi\
```

## Environment actually used

| Item | Value |
|---|---|
| Host | 13th Gen Intel Core i5-1340P, x86_64, Windows 11 |
| Docker | 29.1.2, Linux engine, 16 CPUs |
| Image | `fanyoung/dealii-deeponet:latest` (local digest `sha256:a45baf49...`) |
| Image architecture | **arm64** (`docker image inspect`); container `uname -m` = `aarch64` |
| Execution | ARM aarch64 under QEMU user emulation on the x86_64 host |
| deal.II | 9.7.1 at `/opt/dealii-9.7.1-codim`, C++17 build, MPI ON, PETSc ON |
| MPI | Open MPI 4.1.6 |
| Compiler | GCC 13.3.0, `-std=c++17`, Release, `make -j2` |
| Topology | one container; OpenMPI reports 8 cores × 2 hardware threads |
| Threads | one thread per rank (`OMP_NUM_THREADS=1`, `MPI_InitFinalize(...,1)`) |
| np=8 placement | `--bind-to core`, one rank per core, single node |

`nproc` prints 1 inside the emulated container, but
`getconf _NPROCESSORS_ONLN` = 16, `/proc/cpuinfo` has 16 processors, affinity
is `0-15`, and OpenMPI sees 8 cores/16 hardware threads. The np=8 runs bind
one rank to each of 8 distinct cores. This is a qemu-user artifact, not a
one-CPU container.

## What was fixed

1. **CMake signature conflict (first configure of the corrected build
   failed).** To link MPI explicitly I added
   `target_link_libraries(torus_mg_jacobi PRIVATE MPI::MPI_CXX)`, while
   `deal_ii_setup_target()` uses the plain form internally. CMake prohibits
   mixing them:
   ```
   CMake Error at .../macro_deal_ii_setup_target.cmake:124 (target_link_libraries):
     The keyword signature for target_link_libraries has already been used
     with the target "torus_mg_jacobi". All uses ... must be either
     all-keyword or all-plain.
   ```
   Fixed by using the plain form
   `target_link_libraries(torus_mg_jacobi MPI::MPI_CXX)`. The exact failing
   console transcript is in `logs/configure_initial_failure.log`.
2. **CMake was not using its own `DEAL_II_DIR` cache entry.** The pasted file
   set `DEAL_II_DIR` to the image's C++17 install but did not pass it to
   `find_package`. Added `HINTS "${DEAL_II_DIR}" "$ENV{DEAL_II_DIR}"`,
   explicit C++17, and a guard that rejects the C++20 `/usr/local` deal.II.
   `logs/configure.log` now shows
   `Using the deal.II-9.7.1 installation found at /opt/dealii-9.7.1-codim`.
   `logs/compile_flags.txt` shows `-std=c++17`.
3. **Diagnostics added, no numerical changes:**
   - a serial-reference check of one distributed Jacobi sweep at the finest
     level;
   - per-rank preconditioner-application counts and per-level sweeps;
   - a solution dump in global active-DoF order with support-point
     coordinates;
   - peak RSS per rank via `getrusage(RUSAGE_SELF).ru_maxrss`;
   - rank-0-only setup output.

   The torus PDE, coefficient, 16,384-DoF finest grid, coarse solve,
   intergrid transfers, root-only CG, and damped-Jacobi formula are
   unchanged.

## Experiment preserved

- Torus `R=2`, `r=1`; `GridGenerator::torus`; `TorusManifold`; 5 uniform
  global refinements.
- DoFs by level: **16, 64, 256, 1024, 4096, 16384** (Q1, active DoFs 16384).
- `kappa = 1.1 + sin^2(xi) cos^2(eta)`, reaction `1`, manufactured solution
  `u = sin(xi) sin(eta)`.
- Coarse solve: `MGCoarseGridHouseholder` on the level-0 matrix (16 DoFs).
- Transfers: `MGTransferPrebuilt`; interface operators preserved.
- Root-only outer `SolverCG`; all ranks run the geometric-multigrid
  preconditioner, with the MpiJacobi smoother distributing disjoint rows and
  combining them by `MPI_Allgatherv`.
- No distributed CG, no DeepONet, no AMG.

## Verification results

### Grid and solver
Every run reports exactly 16384 DoFs and levels 16, 64, 256, 1024, 4096,
16384. The 1- and 2-rank required commands exit 0 with **16 CG iterations**
and a recomputed true relative residual of **4.86112450153e-08** (≤ 1e-7).

### Only rank 0 runs CG
`solve_with_multigrid()` calls `solver.solve(...)` only inside the
`if (rank == 0)` branch. During the solve, non-root ranks only execute the
command loop and `mg_preconditioner.vmult(...)`; they never call the CG
solver. Each run prints `outer_cg_owner_rank=0` and a single `iterations=`
line. Every rank reports exactly 16 preconditioner applications:
```
np=1: 16
np=2: 16,16
np=4: 16,16,16,16
np=8: 16,16,16,16,16,16,16,16
```

### Same smoother collectives on all ranks
Per-level Jacobi sweeps are identical on every rank in every run:
level 0 = 0 (coarse direct solve), levels 1–5 = 32 each (16 V-cycles ×
pre+post smoothing). `collective_order_check=PASS` in all 20 scaling runs.
A mismatched collective sequence in the `MPI_Allgatherv`-based smoother
would deadlock rather than produce a count; none of the runs did.

### Jacobi sweep uses the same old iterate
Before the solve, all ranks perform one sweep at level 5 on the same
deterministic `u_old` and `rhs`. Rank 0 compares the gathered distributed
result with the serial reference
`u_old + omega * D^{-1} (rhs - A u_old)`. In every run:
```
jacobi_sweep_check_level=5 max_abs_diff=5.27355936697e-16 PASS
```
The code computes all local residuals from `u` into a local buffer, calls
`MPI_Allgatherv`, and only then writes the new values back into `u`.

### One-rank vs multi-rank solution
The solution is dumped in the global active-DoF ordering with support-point
coordinates. Comparing the final scaling run dumps:

| candidate | compared DoFs | max abs diff | relative L2 diff | max coord diff | tolerance | result |
|---|---:|---:|---:|---:|---:|---|
| np=2 | 16384 | 0 | 0 | 0 | 1e-12 | PASS |
| np=4 | 16384 | 0 | 0 | 0 | 1e-12 | PASS |
| np=8 | 16384 | 0 | 0 | 0 | 1e-12 | PASS |

The solutions are **bitwise identical**, not merely within tolerance. The
earlier saved dumps match the final exact-`mpirun` dumps for np=1/2 byte for
byte, and the np=4/8 scaling-run dumps also match bitwise.
Output: `results/verification_solutions.txt`.

### Error sanity check
The exact-solution errors are identical across ranks:
H1 seminorm 0.0775799825788, L2 0.00240578316152, Linfty 0.000834604550619.

## Scaling (smoke-test data only)

Five independent solves per rank count, one thread per rank, `--bind-to core`,
one rank per core, single node. `preconditioner_s` and `solve_s` are rank-0
wall times; solve time is the outer CG loop only (assembly, setup, the
pre-solve sweep check, residual post-check, and the dump are excluded).

| ranks | runs | median preconditioner [s] | median solve [s] | CG iters | median true residual | speedup solve vs 1 rank | peak RSS max [KiB] |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 0.735690 | 0.970377 | 16 | 4.86112450153e-08 | 1.000 | 247952 |
| 2 | 5 | 0.643018 | 0.853046 | 16 | 4.86112450153e-08 | 1.138 | 248700 |
| 4 | 5 | 1.058191 | 1.338107 | 16 | 4.86112450153e-08 | 0.725 | 249956 |
| 8 | 5 | 2.269999 | 2.710397 | 16 | 4.86112450153e-08 | 0.358 | 249620 |

All 20 runs exited 0; all have 16 CG iterations, residual ≤ 1e-7, and both
the collective and Jacobi-sweep checks PASS. Raw rows:
`results/scaling.csv`; medians: `results/scaling_summary.csv`.

The exact required `mpirun` runs are also in the logs:
| run | preconditioner_s | solve_s | true residual | peak RSS [KiB] |
|---|---:|---:|---:|---|
| `mpirun -np 1` | 0.576456277 | 0.756213294 | 4.86112450153e-08 | 245704 |
| `mpirun -np 2` | 0.613693729 | 0.812052785 | 4.86112450153e-08 | 248772 / 248552 |

Equivalent earlier `mpiexec -n 1/2` runs are in `logs/run_np*.{out,err,status}`.

Run-to-run spread under QEMU is large; the np=2 median being slightly faster
than np=1 and the np=4/8 regressions should not be interpreted as algorithmic
scaling. The problem has only 16,384 DoFs, the mesh and level metadata are
replicated on every rank, and every V-cycle uses all-rank `MPI_Allgatherv`
calls.

### Peak memory per rank

`getrusage(RUSAGE_SELF).ru_maxrss` (KiB) measured after the solve, maximum
over the five runs. RSS includes the shared deal.II/MPI library pages, so it
is not exclusive heap usage.

| ranks | peak RSS per rank [KiB] |
|---:|---|
| 1 | 247952 |
| 2 | 248700, 247788 |
| 4 | 249956, 248764, 248836, 248732 |
| 8 | 249620, 249196, 248960, 248848, 249220, 248784, 248868, 248704 |

Full table: `results/peak_memory_by_rank.csv`.

## Exact commands

Container (the image is ARM; `--platform linux/amd64` still runs `aarch64`):

```powershell
docker run -d --name torus-mg --platform linux/amd64 `
  -v "D:\MS\Munster\ms:/workspace" -w /workspace/torus_mg_jacobi `
  fanyoung/dealii-deeponet:latest sleep infinity
docker exec -it torus-mg bash
```

Build, from the project root inside the container:

```bash
cd /workspace/torus_mg_jacobi
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j2
```

Required runs:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mpirun --allow-run-as-root -np 1 ./torus_mg_jacobi 1 6
mpirun --allow-run-as-root -np 2 ./torus_mg_jacobi 1 6
```

Scaling (five independent runs each; the script is in the project root):

```bash
bash ../run_scaling.sh
```

Analysis (from the project root):

```bash
./tools/collect_scaling.sh
./tools/summarize_scaling.sh results/scaling.csv > results/scaling_summary.csv
awk -F, -f tools/peak_memory.awk results/scaling.csv > results/peak_memory_by_rank.csv
awk -v tol=1e-12 -f tools/compare_solutions.awk \
    build/solution-rank0-np1.txt build/solution-rank0-np2.txt
```

## Files and logs

| Path | Contents |
|---|---|
| `src/main.cpp` | corrected solver plus diagnostics |
| `CMakeLists.txt` | C++17, C++20-install guard, MPI link, deal.II HINTS |
| `logs/configure.log` | successful configure, deal.II `/opt` |
| `logs/configure_initial_failure.log` | first CMake failure transcript |
| `logs/build.log` | final incremental build (compile + link) |
| `logs/compile_flags.txt` | `-std=c++17` flags and full link line |
| `logs/run_np1.out/.err/.status` | equivalent `mpiexec -n 1` run |
| `logs/run_np2.out/.err/.status` | equivalent `mpiexec -n 2` run |
| `logs/run_mpirun_np1.out/.err/.status` | exact required `mpirun -np 1` run |
| `logs/run_mpirun_np2.out/.err/.status` | exact required `mpirun -np 2` run |
| `logs/scale_np{1,2,4,8}_run{1..5}.out/.err/.status` | 20 scaling runs |
| `logs/environment.txt` | arch, CPU, toolchain, deal.II, flags, linkage |
| `logs/image_platform_note.txt` | ARM64/QEMU audit and CPU-count quirk |
| `logs/binding.txt` | np=8 rank-to-core binding |
| `results/scaling.csv` | 20 raw runs: preconditioner/solve time, iterations, residual, peak RSS per rank, checks |
| `results/scaling_summary.csv` | medians and speedup |
| `results/peak_memory_by_rank.csv` | peak RSS per rank |
| `results/solution_np1.txt`, `solution_np2.txt` | verified solution dumps with coordinates |
| `results/verification_solutions.txt` | bitwise 1-vs-N comparison |
| `results/verification_summary.txt` | full check transcript |
| `run_scaling.sh` | 5-run scaling driver |
| `tools/` | comparison, collection, and summary scripts |

## Check status

| Check | Status |
|---|---|
| Build in `fanyoung/dealii-deeponet:latest` | PASS after the CMake link fix |
| Exactly 16,384 DoFs, levels 16..16384 | PASS |
| Only rank 0 runs CG | PASS (code branch + one `iterations` line; non-root preconditioner-only) |
| All ranks in smoother collectives, same counts | PASS (16 calls; 0/32/32/32/32/32 sweeps per level on every rank) |
| Jacobi sweep reads old iterate, combines correctly | PASS (serial-reference max diff 5.27e-16) |
| True relative residual ≤ 1e-7, 1 and 2 ranks | PASS (4.86112450153e-08 both) |
| 1- vs 2-rank solution agreement | PASS (bitwise identical, 16384 DoFs, coords exact) |
| 1- vs 4/8-rank solution agreement (extra) | PASS (bitwise identical) |
| 5 independent solves at 1,2,4,8 ranks | PASS as smoke tests (20/20 exit 0) |
| Native non-emulated performance | NOT RUN (image has no native amd64 content) |
| 16 ranks / multi-node | NOT RUN (not requested; single node only) |

The correctness results are independent of emulation. The timings are not.
