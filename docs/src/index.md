# MiddleOutProteinDesign

`MiddleOutProteinDesign.jl` is the stripped-down Flowception protein design package extracted from the current BranchChain Flowception path.

It keeps only the code required for:

- loading the pretrained `branchchain_feat64.jld` base model,
- lifting it into `BranchChainFlowceptionV3`,
- Flowception training / resume training,
- checkpoint sampling and trajectory export,
- reveal-order visualization.

The key entry points are:

- `MiddleOutProteinDesign.BranchChainFlowceptionV3`
- `MiddleOutProteinDesign.P_flowception`
- `MiddleOutProteinDesign.training_prep_flowception`
- `MiddleOutProteinDesign.design`
- `MiddleOutProteinDesign.load_model`

Scripts live in `scripts/`.
