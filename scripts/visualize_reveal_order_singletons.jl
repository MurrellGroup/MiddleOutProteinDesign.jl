using Pkg
const PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))
const FLOW_ROOT = abspath(joinpath(PROJECT_ROOT, ".."))
const BRANCHINGFLOWS_PATH = joinpath(FLOW_ROOT, "BranchingFlows-component-cmask")
Pkg.activate(PROJECT_ROOT)
Pkg.develop(path = BRANCHINGFLOWS_PATH)

ENV["CUDA_VISIBLE_DEVICES"] = ""

using Random
using Dates
using MiddleOutProteinDesign
using BranchingFlows
using Flowfusion
using ForwardBackward: ContinuousState, ManifoldState, DiscreteState, tensor
using DLProteinFormats: load, PDBSimpleFlatV2, pdbid_clean, unflatten
using GLMakie
using ProtPlot
using ProteinChains
using ProteinChains: ProteinStructure, ProteinChain, Atom, writepdb

const NSAMPLES = parse(Int, get(ENV, "BRANCHCHAIN_REVEAL_ORDER_NSAMPLES", "10"))
const NSTART = parse(Int, get(ENV, "BRANCHCHAIN_REVEAL_ORDER_NSTART", "2"))
const MAXLEN = parse(Int, get(ENV, "BRANCHCHAIN_REVEAL_ORDER_MAXLEN", "1000"))
const REVEAL_TEMPERATURE = haskey(ENV, "BRANCHCHAIN_REVEAL_ORDER_TEMPERATURE") ?
    parse(Float32, ENV["BRANCHCHAIN_REVEAL_ORDER_TEMPERATURE"]) :
    nothing
const OUTDIR = get(
    ENV,
    "BRANCHCHAIN_REVEAL_ORDER_OUTDIR",
    joinpath(
        PROJECT_ROOT,
        "runs",
        "reveal_order_singletons_$(Dates.format(now(), "yyyy-mm-dd_HHMMSS"))",
    ),
)
const RNG_SEED = parse(Int, get(ENV, "BRANCHCHAIN_REVEAL_ORDER_SEED", "1234"))

function reveal_policy(P::DirectionalFlowceptionFlow)
    policy = P.reveal_order
    if isnothing(REVEAL_TEMPERATURE)
        return policy
    end
    policy isa SeededRevealOrder || error(
        "Temperature override requires `SeededRevealOrder`; got $(typeof(policy)).",
    )
    return SeededRevealOrder(
        temperature = REVEAL_TEMPERATURE,
        seed_priority = policy.seed_priority,
        reveal_priority = policy.reveal_priority,
    )
end

function frame_state_from_record(rec, inds)
    locs = ContinuousState(rec.locs[:, :, inds])
    rots = ManifoldState(MiddleOutProteinDesign.rotM, eachslice(rec.rots[:, :, inds], dims = 3))
    aas = DiscreteState(21, rec.AAs[inds])
    return (locs, rots, aas)
end

function protein_from_record_subset(rec, inds; name = "RevealOrder")
    isempty(inds) && return ProteinStructure(name, Atom{Float32}[], ProteinChain{Float32}[])
    samp = frame_state_from_record(rec, inds)
    chains = unflatten(tensor(samp[1]), tensor(samp[2]), tensor(samp[3]), rec.chainids[inds], rec.resinds[inds])[1]
    chain_vec = chains isa ProteinChain ? [chains] : chains
    return ProteinStructure(name, Atom{eltype(tensor(samp[1]))}[], chain_vec)
end

function export_reveal_frame(path, rec, visible)
    inds = findall(visible)
    isempty(inds) && error("Cannot export a reveal-order frame with zero visible residues.")
    writepdb(path, protein_from_record_subset(rec, inds))
end

function reveal_events(P::DirectionalFlowceptionFlow, X1::FlowceptionState, nstart::Int, ::Type{T} = Float32) where T
    groups = vec(X1.groupings)
    target_flow = vec(X1.flowmask)
    target_pad = vec(X1.padmask)
    policy = reveal_policy(P)
    policy isa SeededRevealOrder || error("This visualization expects `SeededRevealOrder`; got $(typeof(policy)).")
    horizon = BranchingFlows.flowception_insertion_horizon(P, T)

    visible0 = target_pad .& .!target_flow
    events = NamedTuple{(:time, :ordinal, :index), Tuple{T, Int, Int}}[]
    ordinal = 0

    start = firstindex(groups)
    while start <= length(groups)
        if !target_pad[start]
            start += 1
            continue
        end
        stop = BranchingFlows.group_segment_stop(groups, target_pad, start)
        seg = start:stop
        local_flow = target_flow[seg]
        fixed_local = findall(.!local_flow)
        design_local = findall(local_flow)

        if !isempty(design_local)
            if isempty(fixed_local) && nstart <= 0
                error("Group $(groups[start]) has no fixed context and `nstart=0` under `SeededRevealOrder`.")
            end
            seed_local = BranchingFlows.choose_seed_positions(X1, seg, design_local, fixed_local, nstart, policy, T)
            for local_idx in seed_local
                ordinal += 1
                push!(events, (; time = zero(T), ordinal, index = first(seg) - 1 + local_idx))
            end

            remaining_local = setdiff(design_local, seed_local)
            if !isempty(remaining_local)
                reveal_local = BranchingFlows.choose_reveal_positions(X1, seg, design_local, fixed_local, seed_local, policy, T)
                reveal_levels = sort!(rand(T, length(reveal_local)))
                for (rank, local_idx) in enumerate(reveal_local)
                    ordinal += 1
                    delay = horizon * T(P.scheduler_inverse(reveal_levels[rank]))
                    push!(events, (; time = delay, ordinal, index = first(seg) - 1 + local_idx))
                end
            end
        end
        start = stop + 1
    end

    sort!(events; by = x -> (x.time, x.ordinal))
    return visible0, events
end

function backbone_only!(struc)
    for chain in struc
        for atoms in chain.atoms
            deleteat!(atoms, 4:length(atoms))
        end
    end
    return struc
end

function empty_structure()
    ProteinStructure("Empty", Atom{Float64}[], ProteinChain{Float64}[])
end

function read_structure_or_empty(path)
    if isfile(path) && filesize(path) > 0
        return backbone_only!(read(path, ProteinStructure))
    end
    return empty_structure()
end

function structure_bounds(structures)
    mins = fill(typemax(Float32), 3)
    maxs = fill(typemin(Float32), 3)
    for s in structures
        for chain in s
            bb = ProteinChains.get_backbone(chain)
            ex = extrema(bb; dims = (2, 3))
            mins = min.(mins, first.(ex))
            maxs = max.(maxs, last.(ex))
        end
    end
    return (mins[1], maxs[1], mins[2], maxs[2], mins[3], maxs[3])
end

function render_mp4(frames_dir::String, out_mp4::String)
    frame_files = sort(filter(f -> endswith(f, ".pdb"), readdir(frames_dir, join = true)))
    structures = read_structure_or_empty.(frame_files)
    bounds = structure_bounds(structures)

    ProtPlot.set_theme!(ProtPlot.theme_black())
    step = GLMakie.Observable(1)
    current = @lift structures[clamp($step, 1, length(structures))]
    fig = GLMakie.Figure(; size = (1280, 720), figure_padding = 1)
    ax = GLMakie.Axis3(fig[1, 1], perspectiveness = 0.2, protrusions = (0, 0, 0, 0), aspect = :data, viewmode = :fit,
        width = 1000, height = 1000, tellwidth = false, tellheight = false)
    ax.azimuth[] = -0.2
    ax.elevation[] = 0.0
    GLMakie.Label(fig[1, 1, Top()], "reveal order", valign = :bottom, font = :bold, fontsize = 18, padding = (0, 0, 0, 0))
    atomplot!(ax, current; color = Dict{String, Symbol}(), default_color = :gray85, show_bonds = true, bond_width = 0.2f0)
    hidespines!(ax)
    hidedecorations!(ax)
    limits!(ax, bounds...)
    GLMakie.record(fig, out_mp4, 1:length(structures); framerate = 24) do t
        step[] = t
        ax.azimuth[] += 0.02f0
    end
end

function eligible_indices(dat)
    return findall(dat.len .< MAXLEN)
end

function sample_examples(dat, nsamples::Int)
    rng = MersenneTwister(RNG_SEED)
    inds = shuffle(rng, eligible_indices(dat))
    chosen = Tuple{Int, Any, Any}[]
    for idx in inds
        rec = dat[idx]
        X1 = MiddleOutProteinDesign.compoundstate(MiddleOutProteinDesign.P_flowception, rec)[1]
        count(vec(X1.flowmask)) == 0 && continue
        push!(chosen, (idx, rec, X1))
        length(chosen) == nsamples && break
    end
    length(chosen) == nsamples || error("Only found $(length(chosen)) eligible masked examples; wanted $nsamples.")
    return chosen
end

function write_example(example_id::Int, idx::Int, rec, X1, outdir::String)
    visible0, events = reveal_events(MiddleOutProteinDesign.P_flowception, X1, NSTART, Float32)
    exdir = joinpath(outdir, "example_$(lpad(example_id, 2, '0'))")
    frames_dir = joinpath(exdir, "frames")
    mkpath(frames_dir)

    current = copy(visible0)
    frame_no = 1
    if any(current)
        export_reveal_frame(joinpath(frames_dir, string(frame_no, pad = 4) * ".pdb"), rec, current)
        frame_no += 1
    end
    for event in events
        current[event.index] = true
        export_reveal_frame(joinpath(frames_dir, string(frame_no, pad = 4) * ".pdb"), rec, current)
        frame_no += 1
    end

    out_mp4 = joinpath(exdir, "example_$(lpad(example_id, 2, '0'))_reveal_order.mp4")
    render_mp4(frames_dir, out_mp4)

    return (
        idx = idx,
        pdb = pdbid_clean(rec.name),
        total_len = length(rec.AAs),
        nfixed = count(visible0),
        ndesign = length(events),
        frames_dir = frames_dir,
        mp4 = out_mp4,
    )
end

function main()
    println("outdir=$(OUTDIR)")
    println("nsamples=$(NSAMPLES) nstart=$(NSTART) maxlen=$(MAXLEN) seed=$(RNG_SEED) reveal_temperature=$(something(REVEAL_TEMPERATURE, MiddleOutProteinDesign.P_flowception.reveal_order.temperature))")
    mkpath(OUTDIR)
    Random.seed!(RNG_SEED)

    dat = load(PDBSimpleFlatV2)
    chosen = sample_examples(dat, NSAMPLES)

    open(joinpath(OUTDIR, "summary.csv"), "w") do io
        println(io, "example,dataset_index,pdb,total_len,nfixed,ndesign,frames_dir,mp4")
        for (example_id, (idx, rec, X1)) in enumerate(chosen)
            println("example=$(example_id) dataset_index=$(idx) pdb=$(pdbid_clean(rec.name))")
            row = write_example(example_id, idx, rec, X1, OUTDIR)
            println(io, join([
                example_id,
                row.idx,
                row.pdb,
                row.total_len,
                row.nfixed,
                row.ndesign,
                row.frames_dir,
                row.mp4,
            ], ","))
            flush(io)
        end
    end
end

main()
