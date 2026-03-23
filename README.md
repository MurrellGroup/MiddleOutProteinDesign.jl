# MiddleOutProteinDesign.jl

Adapted from [BranchChain.jl](https://github.com/MurrellGroup/BranchChain.jl).

`MiddleOutProteinDesign.jl` is a protein-design package built around a Flowception-style variable-length generator with controlled reveal order.

Main references:

- Flowception paper: <https://arxiv.org/abs/2512.11438>
- Flowception project page: <https://flowception-meta.github.io/>
- Branching Flows paper: <https://arxiv.org/abs/2511.09465>

## Project goals

This repository targets two modeling goals.

1. Ordered decoding for protein generation.
   Flowception exposes residues over an extended global time axis and gives each residue its own one-unit local denoising trajectory after it appears. This produces a staged generation process with progressively growing context.

2. Interface-first construction for conditional design.
   In binder design, fixed conditioned residues can identify a binding interface or other anchored context. The reveal-order bridge can expose nearby designable residues early and leave the rest of the binder to be scaffolded around that resolved local context.

## Flowception in this repository

Flowception contributes two separate mechanisms:

1. a variable-length jump process
2. a local-time construction

### Variable-length state space

At sampling time the state has the form

`Z_t = (L_t, X_t^{1:L_t}, τ^{1:L_t}, masks, groupings)`

where:

- `L_t` is the current sequence length
- `X_t^{1:L_t}` are the residue states
- `τ^{1:L_t}` are per-residue local times

The dynamics combine:

- continuous / manifold / discrete flow updates on currently present residues
- insertion jumps that increase sequence length

This is a Markov process on a disjoint union of sequence spaces of different lengths. New residues are sampled from the base distribution through `FlowceptionBirthSampler`. In this repository the birth state includes random coordinates, random rotations, a dummy amino-acid token, and an index bookkeeping state.

### Local time

Each residue evolves on its own local clock after it appears. Flowception separates:

- global time `τ_g`, which indexes the overall generation process
- local time `τ_i`, which measures how long residue `i` has been denoising

If residue `i` has reveal / birth delay `d_i`, then

`τ_i = clip(τ_g - d_i, 0, 1)`.

This gives the following behavior:

- residue `i` is absent while `τ_g < d_i`
- residue `i` appears when `τ_g >= d_i`
- residue `i` denoises under the base process while `0 < τ_i < 1`
- residue `i` is frozen once `τ_i = 1`

Local time is the mechanism that lets earlier residues resolve before later residues have finished denoising.

### Extended global time

The Flowception paper uses an extended global horizon such as `τ_g ∈ [0, 2]`. This repository generalizes that to

`τ_g ∈ [0, total_time]`

with the rule:

- insertions are allowed while `τ_g < total_time - 1`
- every residue still receives exactly one unit of local denoising time

For `total_time = 10`, reveal and insertion events are spread over a long global interval, while each residue still follows a local bridge of length `1`.

### State object

The main process state is `FlowceptionState`, which stores:

- the tuple of component states
- `groupings` for chain-aware batching
- `branchmask`
- `flowmask`
- `padmask`
- `local_t`

The relevant semantics are:

- a residue is hidden until its reveal delay has passed
- once revealed, its `local_t` starts at `0`
- its own state stops evolving after `local_t = 1`
- insertions can still happen elsewhere in the sequence after a given residue has frozen

## Bridge law and training objective

Flowception training uses an auxiliary latent reveal / visibility process in the same way that Edit Flows and Branching Flows use auxiliary latent structure processes.

### Auxiliary reveal process

For fixed endpoint `X_1` and global time `τ_g`, many reveal histories are consistent with the partially revealed state. Let `ξ` denote the latent reveal / visibility process. The bridge law is the marginal

`q(X_t | X_1, τ_g) = ∫ q(X_t, ξ | X_1, τ_g) dξ`

or the corresponding sum over `ξ` when the reveal process is discrete.

Training samples `ξ` and uses Monte Carlo to optimize the marginal objective induced by this bridge law.

### Visible-state losses

Given a sampled bridge state `X_t`, the model predicts endpoint quantities for the visible residues:

- translation endpoint targets
- rotation endpoint targets
- amino-acid endpoint targets

These losses are evaluated at residue-wise local times `local_t`, not at a shared global time.

### Insertion supervision

Hidden residues are converted into insertion targets rather than discarded. After sampling `ξ`, the bridge exposes a visible subsequence and leaves hidden residues in the gaps between visible residues. Those hidden residues induce insertion targets.

For one-sided Flowception, a visible element predicts the hidden mass to its right.

For directional Flowception, the model predicts a two-channel insertion head and the bridge produces the corresponding directional targets. The insertion targets are derived from the same sampled auxiliary reveal process `ξ`.

The training objective therefore combines:

- endpoint prediction on visible residues
- insertion prediction for hidden mass in the visible gaps

This is the part of Flowception that makes the process variable-length.

### Training versus inference

The reveal-order distribution is part of the training bridge. Changing that distribution changes the family of bridge states seen during training.

Inference uses the learned sampler alone. The reveal-order bridge is not used at inference time because it conditions on the true endpoint `X_1`.

## Directional Flowception

This repository uses a bidirectional variant of Flowception.

The model predicts a `2`-channel insertion head:

- channel `1`: insertions to the left of a visible residue
- channel `2`: insertions to the right of a visible residue

The model remains token-centric and does not introduce explicit gap tokens into the backbone.

Within each chain/group:

- `right[i]` and `left[i+1]` correspond to the same physical interior gap
- those directional predictions are pooled into one physical gap quantity
- chain boundaries remain separate, so different chains are never merged through a shared gap

This parameterization allows extension in both directions while preserving chain structure.

## Reveal-order bridge used here

The current process `P_flowception` uses a reveal-order distribution during training. That distribution is defined in `src/models.jl` and is based on `SeededRevealOrder(...)`.

Current process-level settings:

- `total_time = 10`
- `reveal_order = SeededRevealOrder(...)`
- `nstart` chosen by the calling script, typically `2`

### Conditioned residues as revealed context

Fixed conditioned residues are treated as already revealed context in the reveal-order bridge. Designable residues near fixed conditioned residues can therefore receive earlier reveal times and earlier local-time progression.

### Seed selection

Within each chain/group:

- if fixed conditioned residues are present, candidate design residues are scored by proximity to that fixed context
- otherwise, the seed score prefers residues near the center of the designable span

The bridge chooses `nstart` seed residues from that score.

### Spatial growth with Gumbel perturbations

After seed selection, the remaining reveal order is sampled by growing from the currently revealed set.

For each chain/group:

- compute distances in the true `X_1` structure
- for each unrevealed design residue, track its best distance to the revealed set
- repeatedly choose the next residue using that spatial score plus Gumbel noise

This produces a stochastic reveal order that favors residues near already revealed residues and residues near fixed conditioned context.

The default package-level reveal temperature is:

- `flowception_reveal_temperature = 0.75`

Training scripts can override that value for resumed runs.

## Bridge construction in this codebase

For one protein / one chain group, a bridge sample is constructed as follows:

1. Sample global time `τ_g ~ Uniform(0, total_time)`.
2. Sample reveal order `ξ`.
3. Convert the reveal order into reveal delays `d_i(ξ)`.
4. Mark residues as visible if they are fixed context or if `τ_g > d_i(ξ)`.
5. Compute `local_t[i] = clip(τ_g - d_i(ξ), 0, 1)` for each visible residue.
6. Bridge the visible residue states under the base process at those local times.
7. Build insertion targets from the hidden residues between visible residues.

The bridge output therefore contains both:

- visible residues with local-time-conditioned endpoint targets
- insertion targets induced by the hidden gaps

## Sampling process

At inference time the learned model defines a hybrid process on variable-length states:

- between jumps, existing residues evolve under the local-time-conditioned base process toward predicted endpoints
- the insertion head determines jump intensities for adding residues
- when an insertion event occurs, the new residue is initialized from `FlowceptionBirthSampler`

This is a learned jump process on a variable-length protein state space.

## Stochastic processes used in `P_flowception`

The package-level process is

```julia
const P_flowception = DirectionalFlowceptionFlow(
    (
        OUBridgeExpVar(100f0, 150f0, 0.000000001f0, dec = -3f0),
        ManifoldProcess(OUBridgeExpVar(100f0, 150f0, 0.000000001f0, dec = -3f0)),
        DistNoisyInterpolatingDiscreteFlow(D1 = Beta(3.0, 1.5)),
        NullProcess(),
    ),
    FlowceptionBirthSampler;
    total_time = 10f0,
    reveal_order = SeededRevealOrder(...),
)
```

### Translations

Translations use:

- `OUBridgeExpVar(100, 150, 1e-9, dec = -3)`

This is the same Ornstein-Uhlenbeck bridge family used in the earlier BranchChain setup. It gives a strongly regularized continuous bridge for Cartesian coordinates with strong mean reversion.

### Rotations

Rotations use:

- `ManifoldProcess(OUBridgeExpVar(...))`

The same OU-style bridge is lifted to the rotation manifold. Protein frames live on `SE(3)`, so the rotation process is defined on the appropriate manifold-valued state space.

### Amino-acid identities

Sequence uses:

- `DistNoisyInterpolatingDiscreteFlow(D1 = Beta(3.0, 1.5))`

This is the discrete flow-matching process for amino-acid identities. The `Beta(3.0, 1.5)` schedule controls how the discrete corruption / interpolation mass evolves over local time.

### Index bookkeeping

The fourth component is:

- `NullProcess()`

This component carries sequence/index alignment information through the variable-length construction.

## Model path kept in this repository

The model path kept here is:

1. load the pretrained `branchchain_feat64.jld` checkpoint
2. deserialize it into `BranchChainV3`
3. lift it into `BranchChainFlowceptionV3`

`BranchChainFlowceptionV3` keeps the BranchChain IPA backbone and feature-conditioning path and changes the variable-length part of the model:

- adds `local_t` conditioning
- adds `branchmask` conditioning
- replaces the one-sided count head with a 2-channel directional insertion head
- trains under the Flowception bridge and Flowception loss

## Scripts

The main scripts are:

- `scripts/train_feat64_masked_flowception.jl`
- `scripts/train_feat64_masked_flowception_resume.jl`
- `scripts/sample_flowception_checkpoint_chainstorm.jl`
- `scripts/visualize_reveal_order_singletons.jl`

## Scope

This repository keeps the code needed for the current MiddleOut Flowception path:

- the base `BranchChainV3` required to deserialize the pretrained checkpoint
- the lifted `BranchChainFlowceptionV3`
- the directional Flowception process and loss path
- the training, resume, sampling, and reveal-visualization scripts

Legacy branching-flow-only models and unrelated code paths are omitted.

## References

- Branching Flows: <https://arxiv.org/abs/2511.09465>
- Flowception: <https://arxiv.org/abs/2512.11438>
