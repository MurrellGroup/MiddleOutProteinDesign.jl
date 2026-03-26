function textlog(filepath::String, l; also_print = true)
    open(filepath, "a") do io
        write(io, join(string.(l), ", "))
        write(io, "\n")
    end
    also_print && println(join(string.(l), ", "))
end

function training_prep_flowception(b, dat, feature_func; P::DirectionalFlowceptionFlow = P_flowception, nstart = 2)
    sampled = compoundstate.(Ref(P), dat[b])
    X1s = [s[1] for s in sampled]
    hasnobreaks = [s[2] for s in sampled]
    pdb_ids = [s[3] for s in sampled]
    chain_labels = [s[4] for s in sampled]
    bat = directional_flowception_bridge(P, X1s, Uniform(0f0, P.total_time); nstart = nstart)

    chain_features = broadcast_features(pdb_ids, chain_labels, bat.Xt.groupings, feature_func)
    rotξ = Guide(bat.Xt.state[2], bat.X1anchor[2])
    resinds = similar(bat.Xt.groupings) .= 1:size(bat.Xt.groupings, 1)
    return (;
        t = bat.t,
        chainids = bat.Xt.groupings,
        resinds,
        Xt = bat.Xt,
        hasnobreaks,
        rotξ_target = rotξ,
        X1_locs_target = bat.X1anchor[1],
        X1aas_target = bat.X1anchor[3],
        splits_target = bat.insertions_target,
        chain_features,
    )
end

function step_spec(model::BranchChainFlowceptionV3, pdb_id, chain_labels, feature_func; hook = nothing,
    recycles = 0, vidpath = nothing, printseq = true, device = identity, frameid = [1])
    function mod_wrapper(t, Xₜ; frameid = frameid, recycles = recycles)
        if !isnothing(vidpath)
            export_pdb(joinpath(vidpath, "Xt", "$(string(frameid[1], pad = 4)).pdb"), Xₜ.state, Xₜ.groupings, collect(1:length(Xₜ.groupings)))
        end
        length(tensor(Xₜ.state[3])[:]) > 2000 && error("Chain too long")
        chain_features = broadcast_features([pdb_id], [chain_labels], Xₜ.groupings, feature_func)
        printseq && println(replace(DLProteinFormats.ints_to_aa(tensor(Xₜ.state[3])[:]), "X" => "-"), ":", frameid[1])
        resinds = similar(Xₜ.groupings) .= 1:size(Xₜ.groupings, 1)
        input_bundle = ([t]', Xₜ, Xₜ.groupings, resinds, [true], chain_features) |> device
        sc_frames = nothing
        for _ in 1:recycles
            sc_frames, _, _ = model(input_bundle..., sc_frames = isnothing(sc_frames) ? nothing : device(sc_frames))
            sc_frames = cpu(sc_frames)
        end
        pred = model(input_bundle..., sc_frames = isnothing(sc_frames) ? nothing : device(sc_frames)) |> cpu
        state_pred = (
            ContinuousState(values(translation(pred[1]))),
            ManifoldState(rotM, eachslice(values(linear(pred[1])), dims = (3, 4))),
            pred[2],
            nothing,
        )
        if !isnothing(vidpath)
            export_pdb(joinpath(vidpath, "X1hat", "$(string(frameid[1], pad = 4)).pdb"), (state_pred[1], state_pred[2], Xₜ.state[3]), Xₜ.groupings, collect(1:length(Xₜ.groupings)))
        end
        !isnothing(hook) && hook(Xₜ.groupings, Xₜ.state, state_pred)
        frameid[1] += 1
        return state_pred, pred[3]
    end
    return mod_wrapper
end

function X1_from_pdb(P::DirectionalFlowceptionFlow, pdb_rec, segments_to_mask::Vector{String}; exclude_flatchain_nums = Int[], recenter = false)
    pdb_rec.cluster = 1
    rec = DLProteinFormats.flatten(pdb_rec)
    L = length(rec.AAs)
    flatAA_chars = collect(join(DLProteinFormats.AAs[rec.AAs]))
    for ex in exclude_flatchain_nums
        flatAA_chars[rec.chainids .== ex] .= '!'
    end
    flatAAstring = join(flatAA_chars)
    cmask = falses(length(flatAAstring))
    for segment in segments_to_mask
        matches = findall(segment, flatAAstring)
        isempty(matches) && error("Segment $segment not found in $flatAAstring")
        for match in matches
            cmask[match] .= true
        end
    end
    X1locs = MaskedState(ContinuousState(rec.locs), cmask, cmask)
    if recenter
        X1locs.S.state .-= mean(X1locs.S.state, dims = 3)
    end
    X1rots = MaskedState(ManifoldState(rotM, eachslice(rec.rots, dims = 3)), cmask, cmask)
    X1aas = MaskedState(DiscreteState(21, rec.AAs), cmask, cmask)
    index_state = MaskedState(DiscreteState(0, [1:L;]), cmask, cmask)
    return FlowceptionState((X1locs, X1rots, X1aas, index_state), rec.chainids, flowmask = cmask, branchmask = cmask)
end

X1_from_pdb(P::DirectionalFlowceptionFlow, pdb_rec; kwargs...) = X1_from_pdb(P, pdb_rec, [""]; kwargs...)
X1_from_pdb(pdb_rec; kwargs...) = X1_from_pdb(P_flowception, pdb_rec, [""]; kwargs...)

function design(model::BranchChainFlowceptionV3, X1::FlowceptionState, pdb_id, chain_labels, feature_func;
    t0 = 0f0,
    steps = Float32.(t0:0.01f0:P_flowception.total_time),
    path = nothing,
    vidpath = nothing,
    printseq = true,
    device = identity,
    hook = nothing,
    P::DirectionalFlowceptionFlow = P_flowception,
    nstart = 2,
    recycles = 0)
    if steps isa Number
        steps = collect(range(Float32(t0), Float32(P.total_time), length = Int(steps) + 1))
    end
    bat = directional_flowception_bridge(P, [X1], [Float32(t0)]; nstart = nstart)
    X0 = bat.Xt
    frameid = [1]
    if !isnothing(vidpath)
        mkpath(joinpath(vidpath, "Xt"))
        mkpath(joinpath(vidpath, "X1hat"))
    end
    samp = gen(P, X0, step_spec(model, pdb_id, chain_labels, feature_func; vidpath, printseq, device, frameid, recycles, hook), steps)
    printseq && println(replace(DLProteinFormats.ints_to_aa(tensor(samp.state[3])[:]), "X" => "-"), ":", frameid[1])
    if !isnothing(vidpath)
        export_pdb(joinpath(vidpath, "Xt", "$(string(frameid[1], pad = 4)).pdb"), samp.state, samp.groupings, collect(1:length(samp.groupings)))
        export_pdb(joinpath(vidpath, "X1hat", "$(string(frameid[1], pad = 4)).pdb"), samp.state, samp.groupings, collect(1:length(samp.groupings)))
    end
    !isnothing(path) && export_pdb(path, samp.state, samp.groupings, collect(1:length(samp.groupings)))
    return samp
end

design(model, X1; kwargs...) = design(model, X1, "", [""], (x...) -> Dict(); kwargs...)

function gen2prot(samp, chainids, resnums; name = "Gen")
    chain_letters = get.((Dict(zip(0:25, 'A':'Z')),), chainids, 'Z')
    chains = DLProteinFormats.unflatten(tensor(samp[1]), tensor(samp[2]), tensor(samp[3]), chain_letters, resnums)[1]
    return ProteinStructure(name, Atom{eltype(tensor(samp[1]))}[], chains)
end

export_pdb(path, samp, chainids, resnums) = ProteinChains.writepdb(path, gen2prot(samp, chainids, resnums))

step_sched(t) = Float32(1 - (cos(t * pi) + 1) / 2)

function ensure_branchchain_checkpoint_compat!()
    if !isdefined(Main, :BranchChain)
        mod = @__MODULE__
        Core.eval(Main, :(const BranchChain = $mod))
    end
    return nothing
end

function load_model(checkpoint)
    ensure_branchchain_checkpoint_compat!()
    file = hf_hub_download("MurrellLab/BFChainStorm", checkpoint)
    return JLD2.load(file, "model_state")
end

function parse_reveal_target_mode(mode::AbstractString)
    normalized = lowercase(strip(mode))
    if normalized in ("count", "counts", "legacy")
        return CountRevealTarget()
    elseif normalized in ("sparse", "next-slot", "next_slot")
        return SparseRevealTarget()
    elseif normalized in ("rb", "rao-blackwell", "rao_blackwell", "rao-blackwellized", "rao_blackwellized")
        return RaoBlackwellizedRevealTarget()
    end
    error("Unknown reveal target mode `$mode`. Use `count`, `sparse`, or `rb`.")
end

function with_reveal_settings(P::DirectionalFlowceptionFlow; temperature::Union{Nothing, Float32} = nothing, target = nothing)
    ro = P.reveal_order
    ro isa SeededRevealOrder || return P
    tuned_ro = SeededRevealOrder(
        temperature = something(temperature, ro.temperature),
        seed_priority = ro.seed_priority,
        reveal_priority = ro.reveal_priority,
        target = something(target, ro.target),
    )
    return DirectionalFlowceptionFlow(
        P.P,
        P.birth_sampler;
        scheduler = P.scheduler,
        scheduler_derivative = P.scheduler_derivative,
        scheduler_inverse = P.scheduler_inverse,
        split_transform = P.insertion_transform,
        total_time = P.total_time,
        reveal_order = tuned_ro,
    )
end

with_reveal_temperature(P::DirectionalFlowceptionFlow, temperature::Float32) = with_reveal_settings(P; temperature)
