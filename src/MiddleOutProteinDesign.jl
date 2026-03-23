module MiddleOutProteinDesign

using DLProteinFormats
using Flux
using JLD2
using BatchedTransformations
using ProteinChains
using HuggingFaceApi
using BranchingFlows
using Flowfusion
using Distributions
using ForwardBackward
using RandomFeatureMaps
using InvariantPointAttention
using Onion
using StatsBase
using Random

include("design_mask.jl")
include("models.jl")
include("utils.jl")

export BranchChainV3,
    BranchChainFlowceptionV3,
    FlowceptionBirthSampler,
    P_flowception,
    SpatialRevealPriority,
    compoundstate,
    design,
    export_pdb,
    flowception_reveal_temperature,
    gen2prot,
    interface_seed_priority,
    load_model,
    losses,
    residue_positions,
    rotM,
    step_sched,
    step_spec,
    textlog,
    training_prep_flowception,
    with_reveal_temperature,
    X1_from_pdb

function __init__()
    if !isdefined(Main, :BranchChain)
        mod = @__MODULE__
        Core.eval(Main, :(const BranchChain = $mod))
    end
end

end
