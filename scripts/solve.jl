#=
Solve one MPS file with the PDFP feasibility pump.

    julia --project scripts/solve.jl --instance examples/p0201.mps.gz --output_directory results

With `--trajectories K` (K ≥ 2) the script starts K pump processes on the same
GPU, with seeds 100 + seed, 200 + seed, ..., keeps the first feasible solution
and stops the other processes.
=#
using ArgParse
using Printf
using PDFP

function parse_command_line()
    s = ArgParseSettings(description = "PDFP: GPU-accelerated feasibility pump.")
    @add_arg_table! s begin
        "--instance"
        help = "path to the instance, .mps or .mps.gz"
        arg_type = String
        required = true
        "--output_directory"
        help = "directory for the summary and the solution"
        arg_type = String
        default = "results"
        "--time_limit"
        help = "time limit in seconds"
        arg_type = Float64
        default = 1800.0
        "--seed"
        help = "random seed"
        arg_type = Int
        default = 1
        "--trajectories"
        help = "number of concurrent pump processes on the GPU"
        arg_type = Int
        default = 1
        "--batch_policy"
        help = "cost (learn per run whether to try the batch) or always"
        arg_type = String
        default = "cost"
        "--cpu"
        help = "run on the CPU"
        action = :store_true
        "--no_warmup"
        help = "skip the warm-up runs that compile the code before timing"
        action = :store_true
        "--quiet"
        help = "no per-iteration log"
        action = :store_true
    end
    return parse_args(s)
end

instance_name(path) = replace(basename(path), r"\.mps(\.gz)?$" => "")
summary_path(dir, name, seed) = joinpath(dir, "$(name)_seed$(seed).tsv")

const SUMMARY_HEADER = ["instance", "seed", "found", "objective", "max_violation", "iterations",
                        "perturbations", "pdhg_iterations", "root_lp_status", "root_lp_time", "time"]

# Written to a temporary file and renamed, so that a reader never sees a
# partial summary.
function write_summary(path, name, seed, r::PumpResult)
    open(path * ".tmp", "w") do io
        println(io, join(SUMMARY_HEADER, '\t'))
        println(io, join((name, seed, r.found, r.objective, r.max_violation, r.iterations,
                          r.perturbations, r.pdhg_iterations, r.root_lp_status,
                          round(r.root_lp_time, digits = 3), round(r.time, digits = 3)), '\t'))
    end
    mv(path * ".tmp", path; force = true)
end

read_summary(path) = Dict(zip(SUMMARY_HEADER, split(readlines(path)[2], '\t')))

# The seed of the trajectory that found a solution in the least time, if any.
function fastest_solution(dir, name, seeds)
    found = Tuple{Float64,Int}[]
    for seed in seeds
        path = summary_path(dir, name, seed)
        isfile(path) || continue
        s = read_summary(path)
        s["found"] == "true" && push!(found, (parse(Float64, s["time"]), seed))
    end
    return isempty(found) ? nothing : minimum(found)[2]
end

function solve_one(args)
    use_gpu = !args["cpu"]
    if !args["no_warmup"]
        # compile the code on a mixed and a pure-integer instance that need
        # several projections, so that compilation is not timed
        for name in ("pk1", "p0201")
            warmup = read_mps(joinpath(@__DIR__, "..", "examples", "$name.mps.gz"))
            redirect_stdout(devnull) do
                feasibility_pump(warmup; time_limit = 300.0, use_gpu, verbose = !args["quiet"])
            end
        end
    end
    milp = read_mps(args["instance"])
    r = feasibility_pump(milp; time_limit = args["time_limit"], seed = args["seed"], use_gpu,
                         batch_policy = Symbol(args["batch_policy"]), verbose = !args["quiet"])
    name = instance_name(args["instance"])
    dir = args["output_directory"]
    mkpath(dir)
    r.found && write_solution(joinpath(dir, "$(name)_seed$(args["seed"])_solution.tsv"), milp, r.x)
    # the summary is written last: it marks the run as complete
    write_summary(summary_path(dir, name, args["seed"]), name, args["seed"], r)
    return r
end

# Stop a process; a Julia process may not exit on SIGTERM while on the GPU.
function stop(p::Base.Process)
    process_running(p) || return
    kill(p, Base.SIGTERM)
    for _ = 1:50
        process_running(p) || return
        sleep(0.1)
    end
    kill(p, Base.SIGKILL)
end

function solve_concurrent(args)
    name = instance_name(args["instance"])
    dir = args["output_directory"]
    mkpath(dir)
    seeds = [100k + args["seed"] for k = 1:args["trajectories"]]
    flags = [f for f in ("--cpu", "--no_warmup") if args[f[3:end]]]
    procs = map(seeds) do seed
        rm(summary_path(dir, name, seed); force = true)
        cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) $(@__FILE__)
               --instance $(args["instance"]) --output_directory $dir
               --time_limit $(args["time_limit"]) --seed $seed --trajectories 1
               --batch_policy $(args["batch_policy"]) $flags`
        log = joinpath(dir, "$(name)_seed$(seed).log")
        run(pipeline(cmd; stdout = log, stderr = log); wait = false)
    end
    while fastest_solution(dir, name, seeds) === nothing && any(process_running, procs)
        sleep(0.5)
    end
    foreach(stop, procs)
    winner = fastest_solution(dir, name, seeds)
    if winner === nothing
        println("no trajectory found a feasible solution")
    else
        s = read_summary(summary_path(dir, name, winner))
        println("feasible solution from the trajectory with seed $winner: objective $(s["objective"]), ",
                "$(s["iterations"]) iterations, $(s["time"]) s")
    end
end

function main()
    args = parse_command_line()
    args["trajectories"] > 1 ? solve_concurrent(args) : solve_one(args)
    return nothing
end

main()
