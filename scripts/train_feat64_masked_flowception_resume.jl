using Pkg
const PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))
const FLOW_ROOT = abspath(joinpath(PROJECT_ROOT, ".."))
const BRANCHINGFLOWS_PATH = joinpath(FLOW_ROOT, "BranchingFlows-component-cmask")
const ZYGOTE_PATH = joinpath(FLOW_ROOT, "Zygote-jl-1.12-fix")
Pkg.activate(PROJECT_ROOT)
Pkg.develop(path = BRANCHINGFLOWS_PATH)
Pkg.develop(path = ZYGOTE_PATH)

using MiddleOutProteinDesign
using Flux, Distributions, Dates
using DLProteinFormats: load, CHAIN_FEATS_64, PDBSimpleFlatV2, PDBClusters, PDBTable, sample_batched_inds, length2batch, featurizer, broadcast_features, pdbid_clean
using LearningSchedules
using CannotWaitForTheseOptimisers: Muon
using JLD2
using JLD2: jldsave

ENV["CUDA_VISIBLE_DEVICES"] = get(ENV, "CUDA_VISIBLE_DEVICES", "0")
using CUDA, cuDNN
device!(0)
device = gpu

const resume_checkpoint = get(ENV, "BRANCHCHAIN_FLOWCEPTION_RESUME_CHECKPOINT", "")
isempty(resume_checkpoint) && error("Set BRANCHCHAIN_FLOWCEPTION_RESUME_CHECKPOINT to a model_epoch_*_batch_*.jld checkpoint.")

const nstart = parse(Int, get(ENV, "BRANCHCHAIN_FLOWCEPTION_NSTART", "2"))
const max_epochs = parse(Int, get(ENV, "BRANCHCHAIN_FLOWCEPTION_EPOCHS", "15"))
const sample_interval = parse(Int, get(ENV, "BRANCHCHAIN_FLOWCEPTION_SAMPLE_INTERVAL", "5000"))
const l2b_cap = parse(Int, get(ENV, "BRANCHCHAIN_FLOWCEPTION_L2B_CAP", "1500"))
const sample_steps = parse(Int, get(ENV, "BRANCHCHAIN_FLOWCEPTION_SAMPLE_STEPS", "1000"))
const sample_recycles = parse(Int, get(ENV, "BRANCHCHAIN_FLOWCEPTION_SAMPLE_RECYCLES", "2"))
const resume_reveal_temperature = parse(Float32, get(ENV, "BRANCHCHAIN_FLOWCEPTION_RESUME_REVEAL_TEMPERATURE", "10"))
const max_batches = parse(Int, get(ENV, "BRANCHCHAIN_FLOWCEPTION_MAX_BATCHES", "0"))
const insertion_multiplier = parse(Float32, get(ENV, "BRANCHCHAIN_FLOWCEPTION_INSERTION_MULTIPLIER", "0.05"))
const warmdown_epoch = max(max_epochs - 1, 1)

Flux.MLDataDevices.Internal.unsafe_free!(x) = (Flux.fmapstructure(Flux.MLDataDevices.Internal.unsafe_free_internal!, x); return nothing)

struct BatchDataset{T,D,F,P}
    batchinds::T
    dat::D
    train_ff::F
    flow::P
end

Base.length(x::BatchDataset) = length(x.batchinds)
function Base.getindex(x::BatchDataset, i)
    return training_prep_flowception(x.batchinds[i], x.dat, x.train_ff; P = x.flow, nstart = nstart)
end

function batchloader(dat, clusters, len_lbs, train_ff, P_train; device = identity, parallel = true)
    uncapped_l2b = length2batch(l2b_cap, 1.25)
    batchinds = sample_batched_inds(len_lbs, clusters, l2b = x -> min(uncapped_l2b(x), 100))
    @show length(batchinds)
    x = BatchDataset(batchinds, dat, train_ff, P_train)
    dataloader = Flux.DataLoader(x; batchsize = -1, parallel)
    return device(dataloader)
end

function checkpoint_epoch_batch(path::AbstractString)
    stem = splitext(basename(path))[1]
    m = match(r"^model_epoch_(\d+)(?:_batch_(\d+))?$", stem)
    isnothing(m) && error("Checkpoint path must look like model_epoch_<epoch>_batch_<batch>.jld or model_epoch_<epoch>.jld: $path")
    epoch = parse(Int, m.captures[1])
    batch = isnothing(m.captures[2]) ? 0 : parse(Int, m.captures[2])
    return epoch, batch, stem
end

function load_resume_model(checkpoint)
    state = JLD2.load(checkpoint, "model_state")
    model = BranchChainFlowceptionV3(load_model("branchchain_feat64.jld"))
    Flux.loadmodel!(model, state)
    return model
end

function save_checkpoint(path, model, opt_state)
    jldsave(path, model_state = Flux.state(cpu(model)), opt_state = cpu(opt_state))
end

function main()
    resume_epoch, resume_batch, checkpoint_stem = checkpoint_epoch_batch(resume_checkpoint)
    runs_dir = joinpath(PROJECT_ROOT, "runs")
    mkpath(runs_dir)
    rundir = joinpath(runs_dir, "middleout_flowception_resume_$(checkpoint_stem)_$(Date(now()))_$(rand(100000:999999))")
    println("rundir=$(rundir)")
    println(
        "settings checkpoint=$(resume_checkpoint) resume_epoch=$(resume_epoch) resume_batch=$(resume_batch) " *
        "nstart=$(nstart) epochs=$(max_epochs) sample_interval=$(sample_interval) l2b_cap=$(l2b_cap) " *
        "sample_steps=$(sample_steps) sample_recycles=$(sample_recycles) resume_reveal_temperature=$(resume_reveal_temperature) " *
        "max_batches=$(max_batches) insertion_multiplier=$(insertion_multiplier) base_reveal_temperature=$(MiddleOutProteinDesign.flowception_reveal_temperature)"
    )
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
    P_train = with_reveal_temperature(MiddleOutProteinDesign.P_flowception, resume_reveal_temperature)

    println("stage=model")
    model = load_resume_model(resume_checkpoint) |> device

    sched = burnin_learning_schedule(0.00001f0, 0.000250f0, 1.05f0, 0.999995f0)
    opt_state = Flux.setup(Muon(eta = sched.lr, fallback = x -> any(size(x) .== 21)), model)

    textlog("$(rundir)/log.csv", ["epoch", "batch", "learning rate", "loss", "loc_loss", "rot_loss", "aa_loss", "insertion_loss", "insertion_loss_raw", "insertion_multiplier"])

    resumed_batches = 0
    for epoch in resume_epoch:max_epochs
        if epoch == warmdown_epoch
            sched = linear_decay_schedule(sched.lr, 0.000000001f0, 5800)
        end
        for (i, ts) in enumerate(batchloader(dat, clusters, len_lbs, train_ff, P_train; device = device))
            batch_label = epoch == resume_epoch ? resume_batch + i : i
            sc_frames = nothing
            for _ in 1:rand(Poisson(1))
                sc_frames, _, _ = model(ts.t', ts.Xt, ts.chainids, ts.resinds, ts.hasnobreaks, ts.chain_features, sc_frames = sc_frames)
            end
            loss_result, grad = Flux.withgradient(model) do m
                frames, aa_logits, count_log = m(ts.t', ts.Xt, ts.chainids, ts.resinds, ts.hasnobreaks, ts.chain_features, sc_frames = sc_frames)
                l_loc, l_rot, l_aas, l_splits_raw = losses(P_train, (frames, aa_logits, count_log), ts)
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
            textlog("$(rundir)/log.csv", [epoch, batch_label, sched.lr, l, l_loc, l_rot, l_aas, l_splits, l_splits_raw, insertion_multiplier]; also_print = (i == 1 || mod(i, 10) == 0))

            if batch_label >= sample_interval && mod(batch_label, sample_interval) == 0
                for v in 1:3
                    try
                        sampname = "e$(epoch)_b$(batch_label)_samp$(v)"
                        vidpath = "$(rundir)/vids/$(sampname)"
                        template = (MiddleOutProteinDesign.pdb"7F5H"1)[[1, 2]]
                        to_redesign = [template[2].sequence]
                        design(
                            model,
                            X1_from_pdb(P_train, template, to_redesign),
                            template.name,
                            [t.id for t in template],
                            sampling_ff;
                            P = P_train,
                            nstart = nstart,
                            steps = sample_steps,
                            recycles = sample_recycles,
                            vidpath = vidpath,
                            printseq = false,
                            device = device,
                            path = "$(rundir)/samples/$(sampname).pdb",
                        )
                    catch err
                        println("Error in Flowception design sample for samp $v")
                        showerror(stdout, err)
                        println()
                    end
                end
                save_checkpoint("$(rundir)/model_epoch_$(epoch)_batch_$(batch_label).jld", model, opt_state)
            end

            resumed_batches += 1
            if max_batches > 0 && resumed_batches >= max_batches
                save_checkpoint("$(rundir)/model_epoch_$(epoch)_batch_$(batch_label).jld", model, opt_state)
                jldsave("$(rundir)/middleout_flowception_resumed.jld", model_state = cpu(model))
                return
            end
        end
        jldsave("$(rundir)/model_epoch_$(epoch).jld", model_state = Flux.state(cpu(model)), opt_state = cpu(opt_state))
    end
    jldsave("$(rundir)/middleout_flowception_resumed.jld", model_state = cpu(model))
end

main()
