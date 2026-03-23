using Pkg
const PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))
const FLOW_ROOT = abspath(joinpath(PROJECT_ROOT, ".."))
const BRANCHINGFLOWS_PATH = joinpath(FLOW_ROOT, "BranchingFlows-component-cmask")
Pkg.activate(PROJECT_ROOT)
Pkg.develop(path = BRANCHINGFLOWS_PATH)

using MiddleOutProteinDesign
using Flux, Distributions, Dates
using DLProteinFormats: load, CHAIN_FEATS_64, PDBSimpleFlatV2, PDBClusters, PDBTable, sample_batched_inds, length2batch, featurizer, broadcast_features, pdbid_clean
using LearningSchedules
using CannotWaitForTheseOptimisers: Muon
using JLD2: jldsave

ENV["CUDA_VISIBLE_DEVICES"] = get(ENV, "CUDA_VISIBLE_DEVICES", "1")
using CUDA, cuDNN
device!(0)
device = gpu

const nstart = parse(Int, get(ENV, "BRANCHCHAIN_FLOWCEPTION_NSTART", "2"))
const max_epochs = parse(Int, get(ENV, "BRANCHCHAIN_FLOWCEPTION_EPOCHS", "15"))
const thaw_batch = parse(Int, get(ENV, "BRANCHCHAIN_FLOWCEPTION_THAW_BATCH", "2000"))
const sample_interval = parse(Int, get(ENV, "BRANCHCHAIN_FLOWCEPTION_SAMPLE_INTERVAL", "5000"))
const l2b_cap = parse(Int, get(ENV, "BRANCHCHAIN_FLOWCEPTION_L2B_CAP", "1500"))
const sample_steps = parse(Int, get(ENV, "BRANCHCHAIN_FLOWCEPTION_SAMPLE_STEPS", "1000"))
const sample_recycles = parse(Int, get(ENV, "BRANCHCHAIN_FLOWCEPTION_SAMPLE_RECYCLES", "2"))
const max_batches = parse(Int, get(ENV, "BRANCHCHAIN_FLOWCEPTION_MAX_BATCHES", "0"))
const insertion_multiplier = parse(Float32, get(ENV, "BRANCHCHAIN_FLOWCEPTION_INSERTION_MULTIPLIER", "0.025"))
const warmdown_epoch = max(max_epochs - 1, 1)
Flux.MLDataDevices.Internal.unsafe_free!(x) = (Flux.fmapstructure(Flux.MLDataDevices.Internal.unsafe_free_internal!, x); return nothing)

struct BatchDataset{T,D,F}
    batchinds::T
    dat::D
    train_ff::F
end

Base.length(x::BatchDataset) = length(x.batchinds)
function Base.getindex(x::BatchDataset, i)
    return training_prep_flowception(x.batchinds[i], x.dat, x.train_ff; P = MiddleOutProteinDesign.P_flowception, nstart = nstart)
end

function batchloader(dat, clusters, len_lbs, train_ff; device = identity, parallel = true)
    uncapped_l2b = length2batch(l2b_cap, 1.25)
    batchinds = sample_batched_inds(len_lbs, clusters, l2b = x -> min(uncapped_l2b(x), 100))
    @show length(batchinds)
    x = BatchDataset(batchinds, dat, train_ff)
    dataloader = Flux.DataLoader(x; batchsize = -1, parallel)
    return device(dataloader)
end

function main()
    runs_dir = joinpath(PROJECT_ROOT, "runs")
    mkpath(runs_dir)
    rundir = joinpath(runs_dir, "middleout_flowception_$(Date(now()))_$(rand(100000:999999))")
    println("rundir=$(rundir)")
    println("settings nstart=$(nstart) epochs=$(max_epochs) thaw_batch=$(thaw_batch) sample_interval=$(sample_interval) l2b_cap=$(l2b_cap) sample_steps=$(sample_steps) sample_recycles=$(sample_recycles) max_batches=$(max_batches) insertion_multiplier=$(insertion_multiplier) reveal_temperature=$(MiddleOutProteinDesign.flowception_reveal_temperature)")
    mkpath("$(rundir)/samples")
    mkpath("$(rundir)/vids")

    println("stage=data")
    dat = load(PDBSimpleFlatV2)
    feature_table = load(PDBTable)
    pdb_clusters = load(PDBClusters)

    train_ff = featurizer(feature_table, CHAIN_FEATS_64, all_mask_prob = 0.05, feat_mask_prob = Beta(1, 5))
    sampling_ff = featurizer(feature_table, CHAIN_FEATS_64)
    clusters = [pdb_clusters[c] for c in pdbid_clean.(dat.name)]
    len_lbs = dat.len
    P_flow = MiddleOutProteinDesign.P_flowception

    println("stage=model")
    base_model = load_model("branchchain_feat64.jld")
    model = BranchChainFlowceptionV3(base_model) |> device

    sched = burnin_learning_schedule(0.00001f0, 0.000250f0, 1.05f0, 0.999995f0)
    opt_state = Flux.setup(Muon(eta = sched.lr, fallback = x -> any(size(x) .== 21)), model)
    if thaw_batch > 0
        Flux.freeze!(opt_state)
        Flux.thaw!(opt_state.layers.branch_embedder)
        Flux.thaw!(opt_state.layers.local_t_encoding)
        Flux.thaw!(opt_state.layers.count_decoder)
    end

    textlog("$(rundir)/log.csv", ["epoch", "batch", "learning rate", "loss", "loc_loss", "rot_loss", "aa_loss", "insertion_loss", "insertion_loss_raw", "insertion_multiplier"])
    for epoch in 1:max_epochs
        if epoch == warmdown_epoch
            sched = linear_decay_schedule(sched.lr, 0.000000001f0, 5800)
        end
        for (i, ts) in enumerate(batchloader(dat, clusters, len_lbs, train_ff; device = device))
            if epoch == 1 && thaw_batch > 0 && i == thaw_batch
                Flux.thaw!(opt_state)
                sched = burnin_learning_schedule(0.00001f0, 0.000250f0, 1.05f0, 0.999995f0)
            end
            sc_frames = nothing
            for _ in 1:rand(Poisson(1))
                sc_frames, _, _ = model(ts.t', ts.Xt, ts.chainids, ts.resinds, ts.hasnobreaks, ts.chain_features, sc_frames = sc_frames)
            end
            loss_result, grad = Flux.withgradient(model) do m
                frames, aa_logits, count_log = m(ts.t', ts.Xt, ts.chainids, ts.resinds, ts.hasnobreaks, ts.chain_features, sc_frames = sc_frames)
                l_loc, l_rot, l_aas, l_splits_raw = losses(P_flow, (frames, aa_logits, count_log), ts)
                l_splits = l_splits_raw * insertion_multiplier
                (; val = l_loc + l_rot + l_aas + l_splits, l_loc, l_rot, l_aas, l_splits_raw)
            end
            l = loss_result.val
            l_loc = loss_result.l_loc
            l_rot = loss_result.l_rot
            l_aas = loss_result.l_aas
            l_splits_raw = loss_result.l_splits_raw
            l_splits = l_splits_raw * insertion_multiplier
            Flux.update!(opt_state, model, grad[1])
            (mod(i, 10) == 0) && Flux.adjust!(opt_state, next_rate(sched))
            textlog("$(rundir)/log.csv", [epoch, i, sched.lr, l, l_loc, l_rot, l_aas, l_splits, l_splits_raw, insertion_multiplier]; also_print = (i == 1 || mod(i, 10) == 0))
            if i >= sample_interval && mod(i, sample_interval) == 0
                for v in 1:3
                    try
                    sampname = "e$(epoch)_b$(i)_samp$(v)"
                    vidpath = "$(rundir)/vids/$(sampname)"
                    template = (MiddleOutProteinDesign.pdb"7F5H"1)[[1, 2]]
                    to_redesign = [template[2].sequence]
                    design(model, X1_from_pdb(P_flow, template, to_redesign), template.name, [t.id for t in template], sampling_ff;
                           P = P_flow, nstart = nstart, steps = sample_steps,
                           recycles = sample_recycles, vidpath = vidpath, printseq = false, device = device, path = "$(rundir)/samples/$(sampname).pdb")
                    catch err
                        println("Error in Flowception design sample for samp $v")
                        showerror(stdout, err)
                        println()
                    end
                end
                jldsave("$(rundir)/model_epoch_$(epoch)_batch_$(i).jld", model_state = Flux.state(cpu(model)), opt_state = cpu(opt_state))
            end
            if max_batches > 0 && i >= max_batches
                jldsave("$(rundir)/model_epoch_$(epoch)_batch_$(i).jld", model_state = Flux.state(cpu(model)), opt_state = cpu(opt_state))
                jldsave("$(rundir)/middleout_flowception_tuned.jld", model_state = cpu(model))
                return
            end
        end
        jldsave("$(rundir)/model_epoch_$(epoch).jld", model_state = Flux.state(cpu(model)), opt_state = cpu(opt_state))
    end
    jldsave("$(rundir)/middleout_flowception_tuned.jld", model_state = cpu(model))
end

main()
