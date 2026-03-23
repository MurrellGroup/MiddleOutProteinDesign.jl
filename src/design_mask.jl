#This entire file is just to get "nice" random masks to fix different bits of proteins.

"""
    group_mask(numgroups)

Randomly choose a non-empty subset of group indices to keep.

Used internally when grouping chains by similar lengths: draws a random number
of groups in `1:(numgroups-1)` (or `1` if there is only one group) and returns
their indices without replacement.
"""
function group_mask(numgroups)
    num_to_keep = rand(1:max(1,numgroups-1))
    groups_to_keep = sample(1:numgroups, num_to_keep, replace = false)
    return groups_to_keep
end

"""
    rand_sub_mask(chainids)

Draw a random 1D Boolean mask over positions.

The mask is built from a random number of short contiguous segments whose
lengths and locations are sampled from Poisson-distributed spans and random
directions. If no positions were selected, the mask falls back to all `true`.
"""
#Changed since condsegment_v1.jld was trained!
function rand_sub_mask(chainids)
    l = length(chainids)
    mask = falses(l)
    if rand() < 0.5 #Designable segments, tend to be long-ish
        length_scale = 30
        for _ in 1:(1+rand(Poisson(rand()*12)))
            dir = rand([-1,1])
            pos = rand(1:l)
            span = rand(Poisson(rand()*length_scale))
            ordered = minmax(pos, pos + dir*span)
            mask[max(1,ordered[1]):min(l,ordered[2])] .= true
        end
    else #Fixed segments (everything else designable), can be long or short
        mask .= true
        length_scale = rand()*30
        for _ in 1:(1+rand(Poisson(rand()*3)))
            dir = rand([-1,1])
            pos = rand(1:l) 
            span = rand(Poisson(rand()*length_scale))
            ordered = minmax(pos, pos + dir*span)
            mask[max(1,ordered[1]):min(l,ordered[2])] .= false
        end
    end
    if !any(mask)
        mask .= true
    end
    return mask
end

"""
    getmask_boundaries(mask)

Find the boundaries of contiguous `true` runs in a Boolean mask.

Returns two integer vectors `(starts, ends)` giving the first and last indices
for each run of `true` values in `mask`. Used to scale a template mask from
one chain length to another.
"""
function getmask_boundaries(mask)
    starts = Int[]
    ends = Int[]
    i = 1
    n = length(mask)
    while i <= n
        if mask[i]
            s = i
            i += 1
            while i <= n && mask[i]
                i += 1
            end
            push!(starts, s)
            push!(ends, i - 1)
        else
            i += 1
        end
    end
    return starts, ends
end

"""
    rand_segment_mask(chainids)

Draw a random residue-level design mask that tends to keep similar segments
across chains of similar length.

Most of the time it:

- groups chains by (slightly perturbed) length,
- picks some of these groups to keep (`group_mask`),
- samples a random sub-mask on a template chain (`rand_sub_mask`), and
- rescales the masked segments to the other chains in the same group using
  `getmask_boundaries`.

Occasionally it instead chooses individual chains to keep and independently
samples sub-masks within those chains. The result is always non-empty.
"""
function rand_segment_mask(chainids)
    l = length(chainids)
    if rand() < 0.8 
        chain_length_dict = countmap(chainids)
        chain_lengths = collect(values(chain_length_dict))
        perturbed_lengths = chain_lengths .* rand(Uniform(0.99, 1.01), length(chain_lengths))
        perm = sortperm(perturbed_lengths)
        sorted_lengths = perturbed_lengths[perm]
        sorted_chains = collect(keys(chain_length_dict))[perm]
        groups = UnitRange{Int64}[]
        base_ind = 1
        for i in 2:length(sorted_lengths)
            if sorted_lengths[i] - sorted_lengths[base_ind] > 0.05*sorted_lengths[base_ind]
                push!(groups, base_ind:i-1)
                base_ind = i
            end
        end
        push!(groups, base_ind:length(sorted_lengths))
        chain_groups = [sorted_chains[g] for g in groups] #chain_groups is a vector of vectors of chain ids
        chain_group_inds_to_keep = group_mask(length(chain_groups))
        mask = falses(l)
        for cgi in chain_group_inds_to_keep
            chain_group = chain_groups[cgi]
            template_chain = rand(chain_group)
            m = rand_sub_mask(chainids[chainids .== template_chain])
            template_l = length(m)
            mask_boundaries = getmask_boundaries(m)
            for gi in chain_groups[cgi]
                sub_inds = findall(chainids .== gi)
                sub_l = length(sub_inds)
                scaled_lb = Int.(ceil.(sub_l .* mask_boundaries[1] ./ template_l))
                scaled_ub = Int.(floor.(sub_l .* mask_boundaries[2] ./ template_l))
                for ind in 1:length(scaled_lb)
                    mask[sub_inds[scaled_lb[ind]:scaled_ub[ind]]] .= true
                end
            end
        end
        if !any(mask)
            mask .= true
        end
        return mask
    end
    #20%:
    unique_chains = unique(chainids)
    chains_to_keep = group_mask(length(unique_chains))
    mask = falses(l)
    for ci in chains_to_keep
        mask[findall(chainids .== ci)] .= rand_sub_mask(chainids[chainids .== ci])
    end
    if !any(mask)
        mask .= true
    end
    return mask
end

#This version masks entire chains, but the mask is often shared when chains are similar lengths to prevent cheating.
"""
    rand_chain_mask(chainids)

Draw a random chain-level design mask, optionally sharing which chains are
kept across chains of similar length.

Roughly:

- with probability 0.2: keep all chains (`trues`),
- otherwise: group chains by similar length, choose some groups to keep
  (`group_mask`), and set the corresponding chains to `true`,
- falling back to sampling individual chains when no grouping event happens.
"""
function rand_chain_mask(chainids)
    l = length(chainids)
    if rand() < 0.2
        return trues(l)
    end
    #60% of the draws remainder: The case where we mask chains together when they are similar lengths.
    if rand() < 0.75 
        chain_length_dict = countmap(chainids)
        chain_lengths = collect(values(chain_length_dict))
        perturbed_lengths = chain_lengths .* rand(Uniform(0.99, 1.01), length(chain_lengths))
        perm = sortperm(perturbed_lengths)
        sorted_lengths = perturbed_lengths[perm]
        sorted_chains = collect(keys(chain_length_dict))[perm]
        groups = UnitRange{Int64}[]
        base_ind = 1
        for i in 2:length(sorted_lengths)
            if sorted_lengths[i] - sorted_lengths[base_ind] > 0.05*sorted_lengths[base_ind]
                push!(groups, base_ind:i-1)
                base_ind = i
            end
        end
        push!(groups, base_ind:length(sorted_lengths))
        chain_groups = [sorted_chains[g] for g in groups] #chain_groups is a vector of vectors of chain ids
        chain_group_inds_to_keep = group_mask(length(chain_groups))
        mask = falses(l)
        for cgi in chain_group_inds_to_keep
            for gi in chain_groups[cgi]
                mask[chainids .== gi] .= true
            end
        end
        return mask
    end
    #20% of the remainder:
    unique_chains = unique(chainids)
    chains_to_keep = group_mask(length(unique_chains))
    mask = falses(l)
    for ci in chains_to_keep
        mask[chainids .== ci] .= true
    end
    return mask
end

"""
    rand_mask(chainids)

Draw a random design mask for a protein.
- with probability 0.2: keep all chains (`trues`),
- with probability 0.4: sample a random chain-level mask,
- with probability 0.4: sample a random segment-level mask,
"""
function rand_mask(chainids)
    if rand() < 0.2
        return trues(length(chainids))
    end
    if rand() < 0.5
        return rand_chain_mask(chainids)
    end
    return rand_segment_mask(chainids)
end

"""
    nobreaks(resinds, chainids, cmask)

Check that the masked region does not contain internal chain breaks.

Returns `true` if, for every pair of adjacent residues where at least one is
masked and both belong to the same chain, their residue indices differ by
exactly one. Otherwise returns `false`. This is used as a simple quality flag
for sampled design masks.
"""
function nobreaks(resinds, chainids, cmask)
    for i in 1:length(resinds)-1
        if (cmask[i] || cmask[i+1]) && (chainids[i] == chainids[i+1]) && (resinds[i] + 1 != resinds[i+1])
            return false
        end
    end
    return true
end