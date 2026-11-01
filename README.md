# PDFP: GPU-Accelerated Feasibility Pump via PDHG

This repository contains the code of a GPU-accelerated feasibility pump for mixed-integer linear programming, described in the paper

- PDFP: GPU-Accelerated Feasibility Pump via PDHG (Ruofeng Liu, Haihao Lu, Rahul Mazumder)

The feasibility pump looks for a feasible solution of

```math
\min_x \ c^\top x \quad \text{s.t.} \quad A_{\mathrm{ineq}} x \le b_{\mathrm{ineq}}, \quad A_{\mathrm{eq}} x = b_{\mathrm{eq}}, \quad \ell \le x \le u, \quad x_j \in \mathbb{Z} \ (j \in I).
```

It alternates two steps: it rounds an LP-feasible point to an integral point, and it projects the integral point back onto the feasible region of the LP relaxation, in $\ell_1$ distance over the integer variables. It stops when the two points meet. This implementation has three parts.

- **PDFP**, the projector. It solves the pump's $\ell_1$ projection problem directly, with restarted Halpern PDHG with reflection on its saddle-point form, without auxiliary variables or constraints, also for general integers. PDHG is matrix-free: its main work is matrix–vector products, which run in parallel on the GPU. The data stay on the GPU for the whole run, and every projection is warm-started from the previous one.
- **Guarded two-phase CPR**, the rounder. It extends the constraint-propagation rounding (CPR) of FP 2.0: it fixes all variables whose LP value is already integral in a single propagation, and falls back to sequential CPR when this batch makes a constraint infeasible. A cost rule learns, per run, whether trying the batch pays. When the first rounding is far from feasible, a violation-guided rule rounds each variable to the end of its domain that adds the least constraint violation.
- **Concurrent trajectories**. Two pump processes with independent seeds share one GPU; the first solution found is kept.

## Setup

The code needs Julia (tested with 1.12) and, to run on the GPU, an NVIDIA GPU with a recent driver. A one-time step installs the dependencies:

```shell
julia --project -e 'import Pkg; Pkg.instantiate()'
```

## Running

`scripts/solve.jl` runs the pump on an MPS file (`.mps` or `.mps.gz`):

```shell
julia --project scripts/solve.jl --instance examples/p0201.mps.gz --output_directory results
```

```
PDFP feasibility pump: p0201
model       133 rows, 201 columns (201 integer, 0 continuous), 1923 nonzeros
device      NVIDIA H100 80GB HBM3
root LP     OPTIMAL after 600 iterations, 0.03 s
rounder     all schedule, violation-guided rounding off (first rounding violates by 9)
  iter     distance  violation       pdhg   gap tol   action      time
     1      8.08613          1        600  1.00e-02    round     0.06s
     2      1.23237          5        400  9.80e-03    round     0.08s
     3     0.116077          5        200  9.60e-03     flip     0.09s
     4      6.22643         39        400  9.41e-03    cycle     0.10s
     5      32.6474         83        400  9.22e-03    round     0.12s
FEASIBLE    objective 11020 after 5 iterations, 0.12 s
            root LP 0.03 s, projections 0.09 s, rounding 0.00 s
```

Each line of the log is one pump iteration: the $\ell_1$ distance of the projection to the rounded point, the largest constraint violation of the rounded point, the PDHG iterations of the projection, the gap tolerance it was solved to, and what produced the next rounded point (`round`; `flip`, FP's move when the rounding repeats; `cycle` or `restart`, a random perturbation after a cycle or every 100 iterations; `integral`, the projection itself is a solution).

The script writes `results/p0201_seed1.tsv`, a one-line summary, and, if a solution was found, `results/p0201_seed1_solution.tsv`, with one `name<TAB>value` line per variable. Before the timed run it runs the pump on two small instances, so that compilation is not timed. Its options are:

| option | default | description |
| --- | --- | --- |
| `--instance` | | path to the `.mps` or `.mps.gz` file |
| `--output_directory` | `results` | directory for the summary and the solution |
| `--time_limit` | 1800 | time limit in seconds |
| `--seed` | 1 | random seed |
| `--trajectories` | 1 | number of concurrent pump processes on the GPU |
| `--batch_policy` | `cost` | `cost`: learn per run whether to try the batch; `always`: try it on every rounding |
| `--cpu` | | run on the CPU |
| `--no_warmup` | | skip the warm-up runs |
| `--quiet` | | no per-iteration log |

With `--trajectories 2`, the script starts two pump processes on the same GPU, with seeds `100 + seed` and `200 + seed`. It stops both as soon as one of them has found a solution, and reports that one.

## Julia interface

```julia
using PDFP

milp = read_mps("examples/p0201.mps.gz")
result = feasibility_pump(milp; time_limit = 600.0, seed = 1, verbose = true)

result.found                 # true if a feasible solution was found
result.x                     # the solution
result.objective             # c'x + c0
is_feasible(milp, result.x)  # integrality, bounds and constraints to 1e-6
```

`feasibility_pump` returns a `PumpResult`, which also reports the number of iterations, restarts and PDHG iterations, and the time spent in the root LP, the projections and the rounding. The first call compiles the code; for timing, call the function once on a small instance first.

## Parameters

The keyword arguments of `feasibility_pump` and their defaults, which are the settings of the paper's experiments:

| name | default | description |
| --- | --- | --- |
| `time_limit` | 1800.0 | time limit in seconds, root LP included |
| `seed` | 1 | seed for the random choices of the rounding and the restarts |
| `use_gpu` | `CUDA.functional()` | run the PDHG solvers on the GPU |
| `verbose` | `false` | print the log |
| `flip_parameter` | 20 | FP's $T$: when the rounding repeats, flip a random number of variables between $T/2$ and $3T/2$ |
| `restart_period` | 100 | FP's $R$: perturb at least once every $R$ iterations |
| `swap_iteration` | 10 | from this iteration on, CPR swaps $\lfloor \lvert I\rvert/10 \rfloor$ random pairs of its ranking |
| `gap_start` | 1e-2 | gap tolerance of the first projection |
| `gap_decay` | 0.98 | factor applied to the gap tolerance after each projection |
| `gap_floor` | 1e-4 | smallest gap tolerance |
| `residual_ratio` | 0.1 | ratio of the residual tolerance to the gap tolerance |
| `root_lp_tolerance` | 1e-2 | relative tolerance of the root LP relaxation, solved by PDLP |
| `projection_iteration_limit` | 500000 | PDHG iterations per projection |
| `batch_policy` | `:cost` | `:cost` learns per run whether trying the batch pays; `:always` tries it on every rounding |
| `push_threshold` | 0.01 | if fewer than this fraction of the LP-integral variables are pushed against a bound by the root LP's objective, the rounder first rounds the fractional variables that the random swaps moved ahead of the batch |
| `violation_threshold` | 5000 | the violation-guided rule is on if the first rounding violates the constraints without continuous variables by at least this much in total |
| `propagation_budget` | 100000 | rows propagated per propagation call on pure-integer models (no limit on mixed-integer ones) |
| `feasibility_tolerance` | 1e-6 | absolute tolerance on integrality and on every constraint |

On models with continuous variables, the success test reads the continuous values of the projection, so the projection must be accurate in original units. There the solvers measure the primal residual in original units, and a projection that is integral but whose rounding is infeasible is solved again with a tenfold tighter tolerance, up to five times per run, as in the FP 2.0 reference implementation of Salvagnin and Mexi (FeasPumpCollection).

## Code

| file | content |
| --- | --- |
| `src/pump.jl` | the feasibility pump loop |
| `src/pdfp.jl`, `src/pdfp_kernels.jl` | PDFP, the projector, and its CUDA kernels |
| `src/pdlp.jl`, `src/pdlp_kernels.jl` | restarted Halpern PDHG for the root LP relaxation, and its CUDA kernels |
| `src/preconditioner.jl` | Ruiz and Pock–Chambolle scaling, power iteration for $\lVert A\rVert_2$ |
| `src/propagation.jl` | bound propagation over the linear constraints |
| `src/rounding.jl` | CPR, guarded two-phase CPR, violation-guided rounding and the cost rule for the batch |
| `src/milp.jl` | the model, the MPS reader and the feasibility check |

## Reproducing the experiments

The paper evaluates the pump on the 142 instances of MIPLIB 2017 with at least $10^6$ nonzeros listed in `benchmark/instances.txt`, presolved once with SCIP 10.0.3 using the trivial presolving settings of MIPLIB 2017, with symmetry handling and dual reductions turned off. To presolve an instance:

```shell
julia --project=benchmark/presolve -e 'import Pkg; Pkg.instantiate()'
julia --project=benchmark/presolve benchmark/presolve/presolve.jl <instance.mps.gz> <presolved_directory>
```

`benchmark/run.sh` then runs the pump on every instance with seeds 1 to 4 and a time limit of 1800 seconds, with one process or with two concurrent trajectories:

```shell
bash benchmark/run.sh <presolved_directory> results 1
bash benchmark/run.sh <presolved_directory> results_concurrent 2
```

A run counts as solved if it returns, within the time limit, a point that satisfies integrality, the bounds and every constraint of the presolved model to $10^{-6}$; the pump checks this before it reports a solution.

## Citation

If you use this code, please cite:

```bibtex
@article{liu2026pdfp,
  title={{PDFP}: {GPU}-Accelerated Feasibility Pump via {PDHG}},
  author={Liu, Ruofeng and Lu, Haihao and Mazumder, Rahul},
  year={2026}
}
```

## License

PDFP is licensed under the MIT License; see [LICENSE](LICENSE). The instances in `examples/` are from MIPLIB 2017 (https://miplib.zib.de), and `benchmark/presolve/presolv-trivial.set` is from the MIPLIB 2017 feature extractor.
