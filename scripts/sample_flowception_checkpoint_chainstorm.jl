"""
    sample_flowception_checkpoint_chainstorm.jl

Load a MiddleOut Flowception checkpoint, sample protein designs on GPU0, and
export:

- final sample PDBs
- `Xt/` and `X1hat/` trajectory PDB frame directories
- ChainStorm/ProtPlot MP4 trajectory movies

This is the canonical checkpoint visualization script for the Flowception
protein model.
"""

using Pkg
const PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))
const FLOW_ROOT = abspath(joinpath(PROJECT_ROOT, ".."))
const BRANCHINGFLOWS_PATH = joinpath(FLOW_ROOT, "BranchingFlows-component-cmask")
Pkg.activate(PROJECT_ROOT)
Pkg.develop(path = BRANCHINGFLOWS_PATH)

using MiddleOutProteinDesign
using Flux
using CUDA, cuDNN
using DLProteinFormats: load, CHAIN_FEATS_64, PDBTable, featurizer
using JLD2
using Dates
using GLMakie
using ProtPlot

ENV["CUDA_VISIBLE_DEVICES"] = get(ENV, "CUDA_VISIBLE_DEVICES", "0")
device!(0)
device = gpu

const DEFAULT_RUNDIR = joinpath(PROJECT_ROOT, "runs")
const RUNDIR = get(ENV, "BRANCHCHAIN_FLOWCEPTION_SAMPLE_RUNDIR", DEFAULT_RUNDIR)
const NSAMPLES = parse(Int, get(ENV, "BRANCHCHAIN_FLOWCEPTION_SAMPLE_NSAMPLES", "3"))
const NSTEPS = parse(Int, get(ENV, "BRANCHCHAIN_FLOWCEPTION_SAMPLE_STEPS", "1000"))
const RECYCLES = parse(Int, get(ENV, "BRANCHCHAIN_FLOWCEPTION_SAMPLE_RECYCLES", "2"))
const NSTART = parse(Int, get(ENV, "BRANCHCHAIN_FLOWCEPTION_SAMPLE_NSTART", "2"))

function latest_checkpoint(rundir::String)
    files = filter(name -> occursin(r"^model_epoch_\d+_batch_\d+\.jld$", name), readdir(rundir))
    isempty(files) && error("No batch checkpoints found in $rundir")
    parse_key(name) = begin
        m = match(r"^model_epoch_(\d+)_batch_(\d+)\.jld$", name)
        (parse(Int, m.captures[1]), parse(Int, m.captures[2]))
    end
    perm = sortperm(files; by = parse_key)
    return joinpath(rundir, files[last(perm)])
end

const CHECKPOINT = haskey(ENV, "BRANCHCHAIN_FLOWCEPTION_SAMPLE_CHECKPOINT") ?
    ENV["BRANCHCHAIN_FLOWCEPTION_SAMPLE_CHECKPOINT"] :
    latest_checkpoint(RUNDIR)
const CHECKPOINT_STEM = splitext(basename(CHECKPOINT))[1]
const OUTDIR = get(ENV, "BRANCHCHAIN_FLOWCEPTION_SAMPLE_OUTDIR",
    joinpath(RUNDIR, "checkpoint_samples_$(CHECKPOINT_STEM)_r$(RECYCLES)_s$(NSTEPS)"))

function load_checkpoint_model(checkpoint)
    state = JLD2.load(checkpoint, "model_state")
    model = BranchChainFlowceptionV3(load_model("branchchain_feat64.jld"))
    Flux.loadmodel!(model, state)
    return model |> device
end

function render_mp4(vidpath::String, out_mp4::String)
    ProtPlot.animate_trajectory_dir(
        out_mp4,
        [joinpath(vidpath, "Xt"), joinpath(vidpath, "X1hat")];
        labels = ["xₜ", "x̂₁"],
        color_by = [:chain, :chain],
        rotation = 0.02f0,
        end_rotation_speedup = 0.5f0,
        framerate = 24,
        size = (1280, 720),
        theme = :black,
    )
end

function main()
    println("rundir=$(RUNDIR)")
    println("checkpoint=$(CHECKPOINT)")
    println("outdir=$(OUTDIR)")
    println("nsamples=$(NSAMPLES) nsteps=$(NSTEPS) recycles=$(RECYCLES) nstart=$(NSTART)")
    mkpath(OUTDIR)

    feature_table = load(PDBTable)
    sampling_ff = featurizer(feature_table, CHAIN_FEATS_64)
    template = (MiddleOutProteinDesign.pdb"7F5H"1)[[1, 2]]
    to_redesign = [template[2].sequence]

    model = load_checkpoint_model(CHECKPOINT)

    open(joinpath(OUTDIR, "summary.csv"), "w") do io
        println(io, "sample,pdb_path,mp4_path,vid_dir")
        for idx in 1:NSAMPLES
            sample_name = "sample_$(idx)"
            out_pdb = joinpath(OUTDIR, sample_name * ".pdb")
            vidpath = joinpath(OUTDIR, sample_name * "_traj")
            out_mp4 = joinpath(OUTDIR, sample_name * ".mp4")
            println("sampling $(sample_name)")
            design(
                model,
                X1_from_pdb(MiddleOutProteinDesign.P_flowception, template, to_redesign),
                template.name,
                [t.id for t in template],
                sampling_ff;
                P = MiddleOutProteinDesign.P_flowception,
                nstart = NSTART,
                steps = NSTEPS,
                recycles = RECYCLES,
                printseq = false,
                device = device,
                path = out_pdb,
                vidpath = vidpath,
            )
            println("rendering $(sample_name)")
            render_mp4(vidpath, out_mp4)
            println(io, join([sample_name, out_pdb, out_mp4, vidpath], ","))
            flush(io)
            GC.gc(true)
            CUDA.reclaim()
        end
    end
end

main()
