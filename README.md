# MiddleOutProteinDesign.jl

> [!IMPORTANT]
> This repo is for a project that is under active develpoment. Do not expect this to just work if you install it.

Model adapted from [BranchChain.jl](https://github.com/MurrellGroup/BranchChain.jl), but the framework should apply to any model.

`MiddleOutProteinDesign.jl` is a protein-design package built around a Flowception-style variable-length generator with controlled reveal order.

## Required BranchingFlows branch

This repository expects the sibling `BranchingFlows.jl` checkout at

`../BranchingFlows-component-cmask`

to be on branch

`spatial-reveal-order`

because that branch contains the structured Flowception reveal-order targets
used here:

- `CountRevealTarget()`
- `SparseRevealTarget()`
- `RaoBlackwellizedRevealTarget()`

The current training scripts rely on that branch for:

- `SeededRevealOrder(...)`
- the Rao-Blackwellized reveal target default
- the directional structured-target bridge and loss

Main references:

- Flowception paper: <https://arxiv.org/abs/2512.11438>
- Latent Process Generator Matching (to handle Flowception extensions): <https://arxiv.org/abs/2605.20547>
- Branching Flows paper (from original BranchChain repo): <https://arxiv.org/abs/2511.09465>

## Project goals

This repository targets two modeling goals.

1. Ordered decoding for protein generation.
   Flowception exposes residues over an extended global time axis and gives each residue its own one-unit local denoising trajectory after it appears. This produces a staged generation process with progressively growing context.

2. Interface-first construction for conditional design.
   In binder design, fixed conditioned residues can identify a binding interface or other anchored context. The reveal-order bridge can expose nearby designable residues early and leave the rest of the binder to be scaffolded around that resolved local context.

Note: by using a modified Flowception approach we get control over the ordering of element resolution, and we get variable-length generation. If you have architecture constraints that require fixed-length tensors, you can condition on a total length, and you can keep all unmaterialized elements (that are not actively flowing) as "virtual" elements, retaining them in the model pass even though they cannot yet flow.

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

## Model architecture

The trainable model in this repository is `BranchChainFlowceptionV3`.

The model is initialized by:

1. loading the pretrained `branchchain_feat64.jld` checkpoint
2. deserializing that checkpoint into the fixed-length `BranchChainV3` architecture
3. constructing `BranchChainFlowceptionV3(base_model)`

The resulting model reuses the pretrained structure backbone and feature pathway and replaces the variable-length component with a Flowception-specific head and conditioning scheme.

### Residue representation

For each visible residue, the initial token representation is the sum of:

- an amino-acid embedding
- a design-mask embedding
- a chain-break embedding
- a learned embedding of the 64-dimensional conditioning features
- a global-time feature term

The global-time feature term is produced by:

- dividing the scalar global time by `10`
- applying random Fourier features
- projecting the result with `global_t_encoding`

Global time only enters the model through this bottom-level token feature bias.

### Pair representation

Pair features are built from:

- residue indices
- chain identities

These are passed through a pairwise positional encoding, random Fourier features, and a linear projection. The resulting pair tensor is used by all IPA blocks.

### Backbone

The backbone contains:

- `6` IPA blocks
- `6` frame-mover blocks
- self-conditioning cross-frame IPA blocks
- self-conditioning self-frame IPA blocks

The coordinate state is represented as rigid frames with:

- a translation component
- a rotation component

At each block, the model updates residue features with IPA and updates frames with the learned frame movers. The frame-mover time argument is a function of residue-wise `local_t`, so frame evolution is controlled by local time rather than global time.

### Self-conditioning

The model supports recycle-based self-conditioning through `sc_frames`.

Given a current partially revealed state `X_t`:

- the model can first predict frames for that same `X_t`
- those predicted frames are then fed back through the self-conditioning IPA blocks
- the final prediction is made after the requested number of recycle passes

This self-conditioning operates on the current visible state only. It does not try to align frame caches across insertion events.

### Time conditioning

The current time-conditioning split is:

- `local_t` is the strong conditioning signal
- `global_t` is a weak input-feature signal

`local_t` enters in three places:

1. `local_t_encoding(local_t_rff(local_t))` is used as the `cond` input for AdaLN inside the IPA blocks
2. `local_t_feature_encoding(local_t_rff(local_t))` is added to the residue features before the output heads
3. explicit local-time predecoder terms are added before both output heads:
   - `AApre_local_t_encoding(...)`
   - `indelpre_local_t_encoding(...)`

`branchmask` is also embedded and added before the output heads. This tells the model which visible residues are still insertion-active.

### Output heads

The model produces three outputs:

1. rigid frames for translation and rotation
2. amino-acid logits
3. a `2`-channel insertion head

The insertion head is directional:

- channel `1` predicts left insertions
- channel `2` predicts right insertions

The insertion head reads the concatenation of residue activations from backbone blocks `4`, `5`, and `6`. This is the same multi-layer readout pattern used in the original BranchChain count head, but the final head here is a new `2`-channel head rather than the old scalar count head.

There is no deletion head in this model path.

## Training setup

The main training script is:

- `scripts/train_feat64_masked_flowception.jl`

The default settings in that script are:

- `15` epochs
- reveal-order temperature `10`
- `sample_recycles = 2`
- `sample_steps = 1000`
- insertion multiplier `0.05`
- `thaw_batch = 2000`

### Loss

The training loss contains four terms:

- translation loss
- rotation loss
- amino-acid loss
- directional insertion loss

The weighted objective is:

- `20 * l_loc`
- `2 * l_rot`
- `0.1 * l_aas`
- `0.05 * l_insertions_raw`

The translation, rotation, and amino-acid losses use residue-wise local times. The insertion loss is the directional Flowception insertion loss induced by the sampled reveal-order bridge.

There is no deletion loss and no pairwise auxiliary loss in this repository's Flowception path.

### Freeze / thaw schedule

The training script freezes most of the network for the first `2000` batches and thaws only the components that are new or directly affected by the changed conditioning path:

- `feature_embedder`
- `branch_embedder`
- `local_t_encoding`
- `local_t_feature_encoding`
- `AApre_local_t_encoding`
- `indelpre_local_t_encoding`
- `count_decoder`

At batch `2000`, the full model is thawed and the burn-in schedule is restarted.

### Learning-rate schedule

The training script uses:

- `Muon` as the optimizer
- a burn-in learning-rate schedule at the start of training
- the same burn-in schedule again immediately after the thaw point
- a fixed linear warmdown that begins at the penultimate epoch

### Periodic in-training samples

Every `5000` batches, the training script writes:

- sampled final PDBs
- trajectory frame directories under `vids/`
- a model checkpoint

Those periodic samples use:

- the same Flowception process as training
- `recycles = 2`
- `steps = 1000`

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

## Addendum: structured reveal-order insertion loss

Let $x$ denote the current visible state at global time $t$, and let $s$ index the current physical insertion slots of $x$. In directional Flowception, these are:

- the slot before the first visible residue in a group,
- the interior gaps between adjacent visible residues in a group,
- the slot after the last visible residue in a group.

The insertion head predicts rates for these physical slots through the directional left/right parameterization. The independent reveal-order bridge and the structured reveal-order bridge differ in the conditional slot generator that the loss should match.

### Independent reveal order

For the independent bridge, if slot $s$ contains $n_s(x)$ hidden residues of the target, then each of those residues contributes the same reveal hazard. The conditional slot rate has the form

$$
\lambda_s(x, t) = \rho(t)\, n_s(x),
$$

where $\rho(t)$ is the scalar scheduler hazard. The standard Flowception count target is therefore valid, because $n_s(x)$ is a linear parameterization of the conditional generator.

### Structured reveal order

For the structured bridge used here, a latent reveal order is sampled inside each group. Once that latent order has been sampled, the next reveal event in the group is no longer spread uniformly over all hidden residues in the current slot. It is concentrated on the next hidden residue in that latent order.

Let $H_g(x)$ be the hidden residues that remain in group $g$ at state $x$, and let

$$
r_g(x) = |H_g(x)|.
$$

Let $J_g$ denote the next hidden residue in the sampled latent order for group $g$, and let

$$
\mathrm{slot}(j; x)
$$

map a hidden residue $j$ to its current physical insertion slot in $x$.

The current implementation still uses the independent count target. That target does not match the conditional generator induced by the structured bridge.

### Sparse fix

The direct repair is to keep the structured bridge and change the insertion target. Conditional on the sampled latent order, the slot target for group $g$ should be

$$
y^{\mathrm{sparse}}_{g,s}(x, J_g) = r_g(x)\,\mathbf{1}[\,\mathrm{slot}(J_g; x) = s\,].
$$

Summing over groups gives the full target

$$
y^{\mathrm{sparse}}_s(x) = \sum_g y^{\mathrm{sparse}}_{g,s}(x, J_g).
$$

This target places all mass on the current physical slot that contains the next hidden residue in the sampled reveal order, scaled by the number of hidden residues still remaining in that group.

This is the simplest correction. It matches the sampled conditional generator, but it is sparse: each group contributes supervision to only one slot.

### Rao-Blackwellized fix

The current `SeededRevealOrder` implementation samples the next revealed residue in a group by adding independent Gumbel noise to a deterministic score for each remaining hidden residue and taking the minimum. For the current visible state $x$, this gives a tractable conditional distribution over the next revealed residue.

Let $a_g(j; x)$ be the current reveal score for hidden residue $j \in H_g(x)$, and let $\tau$ be the reveal temperature. Then

$$
p_g(j \mid x)
\propto
\exp\!\left(-\frac{a_g(j; x)}{\tau}\right),
\qquad j \in H_g(x),
$$

with the usual zero-temperature limit giving the deterministic argmin rule.

The Rao-Blackwellized target marginalizes over the latent next residue instead of sampling a single one:

$$
y^{\mathrm{RB}}_{g,s}(x) = r_g(x)\sum_{j \in H_g(x)} p_g(j \mid x)\,\mathbf{1}[\,\mathrm{slot}(j; x) = s\,].
$$

and

$$
y^{\mathrm{RB}}_s(x) = \sum_g y^{\mathrm{RB}}_{g,s}(x).
$$

This is the same conditional generator parameter, but averaged over the latent next-residue choice. It is denser and lower-variance than the sparse fix.

In the zero-temperature limit, the Rao-Blackwellized target reduces to the sparse fix.

### Directional parameterization

The model does not predict residue identities. It predicts rates for physical slots through directional token-side heads. The corrected target should therefore be constructed at the level of physical slots, and the loss should compare those slot targets to the physical slot rates implied by the directional pooling:

- left boundary slot: the left head of the first visible residue,
- interior slot: the pooled rate from the right head of residue $i$ and the left head of residue $i+1$,
- right boundary slot: the right head of the last visible residue.

The independent reveal-order case can keep the original count target. The structured reveal-order case should use either the sparse target above or the Rao-Blackwellized target.
