struct NullProcess <: Flowfusion.Process end

Flowfusion.endpoint_conditioned_sample(Xa, Xc, ::NullProcess, t_a, t_b, t_c) = Xa
Flowfusion.step(::NullProcess, Xₜ::Flowfusion.MaskedState, X1targets, s₁, s₂) = Xₜ

oldAdaLN(dim, cond_dim) = AdaLN(Flux.LayerNorm(dim), Dense(cond_dim, dim), Dense(cond_dim, dim))
ipa(layer, frames, x, pair_feats, cond, mask) = layer(frames, x, pair_feats = pair_feats, cond = cond, mask = mask)
crossipa(layer, f1, f2, x, pair_feats, cond, mask) = layer(f1, f2, x, pair_feats = pair_feats, cond = cond, mask = mask)

struct BranchChainV3{L}
    layers::L
end

Flux.@layer BranchChainV3

function BranchChainV3(dim::Int = 384, depth::Int = 6, f_depth::Int = 6; config = nothing)
    layers = (;
        config = config,
        depth = depth,
        f_depth = f_depth,
        mask_embedder = Embedding(2 => dim),
        break_embedder = Embedding(2 => dim),
        t_rff = RandomFourierFeatures(1 => dim, 1f0),
        cond_t_encoding = Dense(dim => dim, bias = false),
        AApre_t_encoding = Dense(dim => dim, bias = false),
        pair_rff = RandomFourierFeatures(2 => 64, 1f0),
        pair_project = Dense(64 => 32, bias = false),
        AA_embedder = Embedding(21 => dim),
        selfcond_crossipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = oldAdaLN(dim, dim)) for _ in 1:depth],
        selfcond_selfipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = oldAdaLN(dim, dim)) for _ in 1:depth],
        ipa_blocks = [IPAblock(dim, IPA(IPA_settings(dim, c_z = 32)), ln1 = oldAdaLN(dim, dim), ln2 = oldAdaLN(dim, dim)) for _ in 1:depth],
        framemovers = [Framemover(dim) for _ in 1:f_depth],
        AAdecoder = Chain(StarGLU(dim, 3dim), Dense(dim => 21, bias = false)),
        indelpre_t_encoding = Dense(dim => 3dim),
        count_decoder = StarGLU(Dense(3dim => 2dim, bias = false), Dense(2dim => 1, bias = false), Dense(3dim => 2dim, bias = false), Flux.swish),
        del_decoder = StarGLU(Dense(3dim => 2dim, bias = false), Dense(2dim => 1, bias = false), Dense(3dim => 2dim, bias = false), Flux.swish),
        feature_embedder = Dense(64 => dim),
    )
    return BranchChainV3(layers)
end

function (fc::BranchChainV3)(t, BSXt, chainids, resinds, breaks, chain_features; sc_frames = nothing)
    l = fc.layers
    Xt = BSXt.state
    cmask = BSXt.flowmask
    pmask = Flux.Zygote.@ignore self_att_padding_mask(BSXt.padmask)
    pre_z = Flux.Zygote.@ignore l.pair_rff(pair_encode(resinds, chainids))
    pair_feats = l.pair_project(pre_z)
    t_rff = Flux.Zygote.@ignore l.t_rff(t)
    cond = reshape(l.cond_t_encoding(t_rff), :, 1, size(t, 2))
    frames = Translation(tensor(Xt[1])) ∘ Rotation(tensor(Xt[2]))
    x = l.AA_embedder(tensor(Xt[3])) .+
        l.mask_embedder(cmask .+ 1) .+
        reshape(l.break_embedder(breaks .+ 1), :, 1, size(t, 2)) .+
        l.feature_embedder(chain_features .+ 0)
    x_a, x_b, x_c = nothing, nothing, nothing
    for i in 1:l.depth
        if sc_frames !== nothing
            x = Flux.Zygote.checkpointed(crossipa, l.selfcond_selfipa[i], sc_frames, sc_frames, x, pair_feats, cond, pmask)
            f1, f2 = mod(i, 2) == 0 ? (frames, sc_frames) : (sc_frames, frames)
            x = Flux.Zygote.checkpointed(crossipa, l.selfcond_crossipa[i], f1, f2, x, pair_feats, cond, pmask)
        end
        x = Flux.Zygote.checkpointed(ipa, l.ipa_blocks[i], frames, x, pair_feats, cond, pmask)
        if i > l.depth - l.f_depth
            frames = l.framemovers[i - l.depth + l.f_depth](frames, x, t = 1 .- (1 .- t .* 0.95f0) .* cmask)
        end
        i == 4 && (x_a = x)
        i == 5 && (x_b = x)
        i == 6 && (x_c = x)
    end
    aa_logits = l.AAdecoder(x .+ reshape(l.AApre_t_encoding(t_rff), :, 1, size(t, 2)))
    catted = vcat(x_a, x_b, x_c)
    indel_pre_t = reshape(l.indelpre_t_encoding(t_rff), :, 1, size(t, 2))
    count_log = reshape(l.count_decoder(catted .+ indel_pre_t, false), :, length(t))
    del_logits = reshape(l.del_decoder(catted .+ indel_pre_t, false), :, length(t))
    return frames, aa_logits, count_log, del_logits
end

struct BranchChainFlowceptionV3{L}
    layers::L
end

Flux.@layer BranchChainFlowceptionV3

function BranchChainFlowceptionV3(dim::Int = 384, depth::Int = 6, f_depth::Int = 6; config = nothing)
    layers = (;
        config = config,
        depth = depth,
        f_depth = f_depth,
        mask_embedder = Embedding(2 => dim),
        branch_embedder = Embedding(2 => dim),
        break_embedder = Embedding(2 => dim),
        t_rff = RandomFourierFeatures(1 => dim, 1f0),
        local_t_rff = RandomFourierFeatures(1 => dim, 1f0),
        global_t_encoding = Dense(dim => dim, bias = false),
        local_t_encoding = Dense(dim => dim, bias = false),
        local_t_feature_encoding = Dense(dim => dim, bias = false),
        AApre_local_t_encoding = Dense(dim => dim, bias = false),
        pair_rff = RandomFourierFeatures(2 => 64, 1f0),
        pair_project = Dense(64 => 32, bias = false),
        AA_embedder = Embedding(21 => dim),
        selfcond_crossipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = oldAdaLN(dim, dim)) for _ in 1:depth],
        selfcond_selfipa = [CrossFrameIPA(dim, IPA(IPA_settings(dim, c_z = 32)), ln = oldAdaLN(dim, dim)) for _ in 1:depth],
        ipa_blocks = [IPAblock(dim, IPA(IPA_settings(dim, c_z = 32)), ln1 = oldAdaLN(dim, dim), ln2 = oldAdaLN(dim, dim)) for _ in 1:depth],
        framemovers = [Framemover(dim) for _ in 1:f_depth],
        AAdecoder = Chain(StarGLU(dim, 3dim), Dense(dim => 21, bias = false)),
        indelpre_local_t_encoding = Dense(dim => 3dim),
        count_decoder = StarGLU(Dense(3dim => 2dim, bias = false), Dense(2dim => 2, bias = false), Dense(3dim => 2dim, bias = false), Flux.swish),
        feature_embedder = Dense(64 => dim),
    )
    layers.local_t_encoding.weight ./= 10
    layers.local_t_feature_encoding.weight ./= 10
    layers.AApre_local_t_encoding.weight ./= 10
    layers.indelpre_local_t_encoding.weight ./= 10
    layers.global_t_encoding.weight ./= 10
    layers.branch_embedder.weight ./= 10
    return BranchChainFlowceptionV3(layers)
end

function BranchChainFlowceptionV3(base::BranchChainV3)
    scaffold = BranchChainFlowceptionV3(; config = base.layers.config)
    shared_layers = (;
        config = base.layers.config,
        depth = base.layers.depth,
        f_depth = base.layers.f_depth,
        mask_embedder = base.layers.mask_embedder,
        break_embedder = base.layers.break_embedder,
        t_rff = base.layers.t_rff,
        global_t_encoding = base.layers.cond_t_encoding,
        pair_rff = base.layers.pair_rff,
        pair_project = base.layers.pair_project,
        AA_embedder = base.layers.AA_embedder,
        selfcond_crossipa = base.layers.selfcond_crossipa,
        selfcond_selfipa = base.layers.selfcond_selfipa,
        ipa_blocks = base.layers.ipa_blocks,
        framemovers = base.layers.framemovers,
        AAdecoder = base.layers.AAdecoder,
        feature_embedder = base.layers.feature_embedder,
    )
    lifted_layers = merge(scaffold.layers, shared_layers)
    return BranchChainFlowceptionV3(lifted_layers)
end

function residue_design_mask(Xt::FlowceptionState)
    loc_state = Xt.state[1]
    return loc_state isa MaskedState ? loc_state.cmask : Xt.flowmask
end

flowception_frame_t(local_t, cmask) = 1 .- (1 .- local_t .* 0.95f0) .* cmask

function (fc::BranchChainFlowceptionV3)(t, Xt::FlowceptionState, chainids, resinds, breaks, chain_features; sc_frames = nothing)
    l = fc.layers
    state = Xt.state
    cmask = residue_design_mask(Xt)
    branchmask = Xt.branchmask
    local_t = reshape(Xt.local_t, 1, size(Xt.local_t)...)
    pmask = Flux.Zygote.@ignore self_att_padding_mask(Xt.padmask)
    pre_z = Flux.Zygote.@ignore l.pair_rff(pair_encode(resinds, chainids))
    pair_feats = l.pair_project(pre_z)
    t_rff = Flux.Zygote.@ignore l.t_rff(t ./ 10)
    local_t_feats = Flux.Zygote.@ignore l.local_t_rff(local_t)
    cond = l.local_t_encoding(local_t_feats)
    global_t_feats = reshape(l.global_t_encoding(t_rff), :, 1, size(t, 2))
    frames = Translation(tensor(state[1])) ∘ Rotation(tensor(state[2]))
    x = l.AA_embedder(tensor(state[3])) .+
        l.mask_embedder(cmask .+ 1) .+
        reshape(l.break_embedder(breaks .+ 1), :, 1, size(t, 2)) .+
        l.feature_embedder(chain_features .+ 0) .+
        global_t_feats
    x_a, x_b, x_c = nothing, nothing, nothing
    for i in 1:l.depth
        if sc_frames !== nothing
            x = Flux.Zygote.checkpointed(crossipa, l.selfcond_selfipa[i], sc_frames, sc_frames, x, pair_feats, cond, pmask)
            f1, f2 = mod(i, 2) == 0 ? (frames, sc_frames) : (sc_frames, frames)
            x = Flux.Zygote.checkpointed(crossipa, l.selfcond_crossipa[i], f1, f2, x, pair_feats, cond, pmask)
        end
        x = Flux.Zygote.checkpointed(ipa, l.ipa_blocks[i], frames, x, pair_feats, cond, pmask)
        if i > l.depth - l.f_depth
            frames = l.framemovers[i - l.depth + l.f_depth](frames, x, t = flowception_frame_t(Xt.local_t, cmask))
        end
        i == 4 && (x_a = x)
        i == 5 && (x_b = x)
        i == 6 && (x_c = x)
    end
    flow_cond = l.branch_embedder(branchmask .+ 1) .+ l.local_t_feature_encoding(local_t_feats)
    x = x .+ flow_cond
    x_a = x_a .+ flow_cond
    x_b = x_b .+ flow_cond
    x_c = x_c .+ flow_cond
    aa_logits = l.AAdecoder(x .+ l.AApre_local_t_encoding(local_t_feats))
    catted = vcat(x_a, x_b, x_c)
    insertion_logits = l.count_decoder(catted .+ l.indelpre_local_t_encoding(local_t_feats), false)
    return frames, aa_logits, insertion_logits
end

const rotM = Flowfusion.Rotations(3)
const flowception_reveal_temperature = parse(Float32, get(ENV, "MIDDLEOUT_FLOWCEPTION_REVEAL_TEMPERATURE", "0.75"))

function default_reveal_seed_scores(design_local::AbstractVector{Int}, ::Type{T}) where {T}
    isempty(design_local) && return T[]
    center = T(first(design_local) + last(design_local)) / T(2)
    return abs.(T.(design_local) .- center) .+ T(1e-4) .* T.(design_local)
end

function residue_positions(X1::FlowceptionState)
    locs = tensor(Flowfusion.unmask(X1.state[1]))
    if ndims(locs) == 2
        return locs
    end
    flat = reshape(locs, size(locs, 1), :, size(locs, ndims(locs)))
    return flat[:, 1, :]
end

function interface_seed_priority(X1::FlowceptionState, seg, design_local, fixed_local, ::Type{T}) where {T}
    isempty(design_local) && return T[]
    fallback = default_reveal_seed_scores(design_local, T)
    fixed_global = findall(vec(X1.padmask) .& .!vec(X1.flowmask))
    isempty(fixed_global) && return fallback

    design_global = first(seg) - 1 .+ design_local
    coords = residue_positions(X1)
    design_coords = coords[:, design_global]
    fixed_coords = coords[:, fixed_global]
    deltas = reshape(design_coords, size(coords, 1), length(design_global), 1) .- reshape(fixed_coords, size(coords, 1), 1, length(fixed_global))
    dists = sqrt.(sum(abs2, deltas; dims = 1))
    contact = dropdims(minimum(dists, dims = 3); dims = (1, 3))
    return T.(contact) .+ fallback
end

function pairwise_euclidean(coords::AbstractMatrix{T}) where {T}
    deltas = reshape(coords, size(coords, 1), size(coords, 2), 1) .- reshape(coords, size(coords, 1), 1, size(coords, 2))
    return sqrt.(dropdims(sum(abs2, deltas; dims = 1); dims = 1))
end

function cross_euclidean(lhs::AbstractMatrix{T}, rhs::AbstractMatrix{T}) where {T}
    deltas = reshape(lhs, size(lhs, 1), size(lhs, 2), 1) .- reshape(rhs, size(rhs, 1), 1, size(rhs, 2))
    return sqrt.(dropdims(sum(abs2, deltas; dims = 1); dims = 1))
end

struct SpatialRevealPriority{F}
    coord_extractor::F
end

SpatialRevealPriority(; coord_extractor = residue_positions) = SpatialRevealPriority(coord_extractor)

function (priority::SpatialRevealPriority)(X1::FlowceptionState, seg, design_local, fixed_local, ::Type{T}) where {T}
    isempty(design_local) && return (; design_design = zeros(T, 0, 0), design_fixed = zeros(T, 0, 0), bias = T[])
    coords = priority.coord_extractor(X1)
    design_global = first(seg) - 1 .+ design_local
    design_coords = T.(coords[:, design_global])
    design_design = pairwise_euclidean(design_coords)

    fixed_global = findall(vec(X1.padmask) .& .!vec(X1.flowmask))
    design_fixed = if isempty(fixed_global)
        zeros(T, length(design_local), 0)
    else
        fixed_coords = T.(coords[:, fixed_global])
        cross_euclidean(design_coords, fixed_coords)
    end
    return (; design_design, design_fixed, bias = default_reveal_seed_scores(design_local, T))
end

FlowceptionBirthSampler(root) = (
    ContinuousState(randn(Float32, 3, 1, 1)),
    ManifoldState(rotM, reshape(Array{Float32}.(Flowfusion.rand(rotM, 1)), 1)),
    DiscreteState(21, [21]),
    DiscreteState(0, [0]),
)

const P_flowception = DirectionalFlowceptionFlow(
    (
        OUBridgeExpVar(100f0, 150f0, 0.000000001f0, dec = -3f0),
        ManifoldProcess(OUBridgeExpVar(100f0, 150f0, 0.000000001f0, dec = -3f0)),
        DistNoisyInterpolatingDiscreteFlow(D1 = Beta(3.0, 1.5)),
        NullProcess(),
    ),
    FlowceptionBirthSampler;
    total_time = 10f0,
    reveal_order = SeededRevealOrder(
        temperature = flowception_reveal_temperature,
        seed_priority = interface_seed_priority,
        reveal_priority = SpatialRevealPriority(),
    ),
)

function compoundstate(P::DirectionalFlowceptionFlow, rec)
    L = length(rec.AAs)
    cmask = rand_mask(rec.chainids)
    breaks = nobreaks(rec.resinds, rec.chainids, cmask)
    X1locs = MaskedState(ContinuousState(rec.locs), cmask, cmask)
    X1rots = MaskedState(ManifoldState(rotM, eachslice(rec.rots, dims = 3)), cmask, cmask)
    X1aas = MaskedState(DiscreteState(21, rec.AAs), cmask, cmask)
    index_state = MaskedState(DiscreteState(0, [1:L;]), cmask, cmask)
    X1 = FlowceptionState((X1locs, X1rots, X1aas, index_state), rec.chainids, flowmask = cmask, branchmask = cmask)
    return X1, breaks, DLProteinFormats.pdbid_clean(rec.name), rec.chain_labels
end

compoundstate(rec) = compoundstate(P_flowception, rec)

function losses(P::DirectionalFlowceptionFlow, X1hat, ts)
    hat_frames, hat_aas, hat_insertions = X1hat
    rotangent = Flowfusion.so3_tangent_coordinates_stack(values(linear(hat_frames)), tensor(ts.Xt.state[2]))
    hat_loc, hat_rot, hat_aas = (values(translation(hat_frames)), rotangent, hat_aas)
    l_loc = floss(P.P[1], hat_loc, ts.X1_locs_target, scalefloss(P.P[1], ts.Xt.local_t, 1, 0.2f0)) * 20
    l_rot = floss(P.P[2], hat_rot, ts.rotξ_target, scalefloss(P.P[2], ts.Xt.local_t, 1, 0.2f0)) * 2
    l_aas = floss(P.P[3], hat_aas, onehot(ts.X1aas_target), scalefloss(P.P[3], ts.Xt.local_t, 1, 0.2f0)) / 10
    insertion_c = scalefloss(P, reshape(ts.t, 1, :), 1, 0.2f0)
    insertion_loss = floss(P, hat_insertions, ts.splits_target, ts.Xt, insertion_c)
    return l_loc, l_rot, l_aas, insertion_loss
end
