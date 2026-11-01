#=
Presolve an instance as in the paper's experiments: SCIP with the trivial
presolving settings of MIPLIB 2017 (presolv-trivial.set, from the MIPLIB 2017
feature extractor), with symmetry handling and dual reductions turned off,
since both can remove feasible solutions.

    julia --project=benchmark/presolve benchmark/presolve/presolve.jl <in.mps[.gz]> <out_directory>

Writes <out_directory>/<name>.mps.
=#
using SCIP
const L = SCIP.LibSCIP

input, output_directory = ARGS
name = replace(basename(input), r"\.mps(\.gz)?$" => "")
settings = joinpath(@__DIR__, "presolv-trivial.set")

scip = Ref{Ptr{L.SCIP}}(C_NULL)
L.SCIPcreate(scip)
S = scip[]
L.SCIPincludeDefaultPlugins(S)
L.SCIPsetMessagehdlrQuiet(S, 1)
# Some MIPLIB instances contain coefficients of 1e20, SCIP's default infinity.
L.SCIPsetRealParam(S, "numerics/infinity", 1e30)

# The settings file was written for an older SCIP. Set the parameters one at a
# time, with the type SCIP reports, and list those that no longer exist.
unknown = String[]
for line in eachline(settings)
    s = strip(line)
    (isempty(s) || startswith(s, "#")) && continue
    key, value = strip.(split(s, "="; limit = 2))
    param = L.SCIPgetParam(S, key)
    param == C_NULL && (push!(unknown, key); continue)
    type = L.SCIPparamGetType(param)
    rc = type == L.SCIP_PARAMTYPE_BOOL    ? L.SCIPsetBoolParam(S, key, value == "TRUE" ? 1 : 0) :
         type == L.SCIP_PARAMTYPE_INT     ? L.SCIPsetIntParam(S, key, parse(Int32, value)) :
         type == L.SCIP_PARAMTYPE_LONGINT ? L.SCIPsetLongintParam(S, key, parse(Int64, value)) :
         type == L.SCIP_PARAMTYPE_REAL    ? L.SCIPsetRealParam(S, key, parse(Float64, value)) :
                                            L.SCIP_OKAY
    rc == L.SCIP_OKAY || push!(unknown, key)
end
isempty(unknown) || @info "parameters not known to this SCIP version" unknown

@assert L.SCIPsetIntParam(S, "misc/usesymmetry", 0) == L.SCIP_OKAY
@assert L.SCIPsetBoolParam(S, "misc/allowstrongdualreds", 0) == L.SCIP_OKAY
@assert L.SCIPsetBoolParam(S, "misc/allowweakdualreds", 0) == L.SCIP_OKAY

L.SCIPreadProb(S, input, C_NULL) == L.SCIP_OKAY || error("could not read $input")
n0, m0 = L.SCIPgetNVars(S), L.SCIPgetNConss(S)
L.SCIPpresolve(S)
n1, m1 = L.SCIPgetNVars(S), L.SCIPgetNConss(S)
mkpath(output_directory)
L.SCIPwriteTransProblem(S, joinpath(output_directory, "$name.mps"), "mps", 0)
println("$name: $n0 variables, $m0 constraints -> $n1 variables, $m1 constraints")
