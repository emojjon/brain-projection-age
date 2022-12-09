import Flux
import Base.iterate

function Base.iterate(p::Flux.Parallel, s = nothing)
    ks = keys(p)
    if isnothing(s)
        k, s = iterate(ks)
    else
        if isnothing(iterate(ks, s))
            return nothing
        else
            k, s = iterate(ks, s)
        end
    end
    return (p[k], s)
end

function changelayers(model, layers, changes; change_chains = false, change_parallel = false, tolerant = false)
    # takes a model or a node therein and applies a collection of changes to a collection of layers (specified as types)
    # recursively in that model or the "submodel" under said node
    # the keyword arguments can be set to true to apply changes to the Flux.Chain or Flux.Parallel layers themselves.
    function mkchng(node, ch)
        # println("Received a change request expressed as a $(typeof(ch)), not complying")
        if !tolerant
            throw(Core.ArgumentError("A $(typeof(ch)) cannot be interpreted as a valid change to apply to a layer"))
        end
    end

    function mkchng(node, ch::Union{Pair, NTuple{2, Any}})
        # println("Applying a change expressed as a $(typeof(ch))")
        setfield!(node, Symbol(ch[1]), typeof(getfield(node, Symbol(ch[1])))(ch[2]))
    end

    function mkchng(node, ch::Function)
        # println("Applying a change expressed as a function")
        ch(node)
    end
    
    function chlayer(node)
        ll = length(changes)
        suf = ll == 1 ? "" : "s"
        # println("Applying $ll change$suf to a $(typeof(node))")
        for ch ∈ changes
            mkchng(node, ch)
        end
    end
    
    function chlayer_recursive(node::Flux.Chain)
        # println("chlayer_recursive, Chain")
        for n ∈ node
            chlayer_recursive(n)
        end
        if change_chains
            chlayer(node)
        end
    end
    
    function chlayer_recursive(node::Flux.Parallel)
        # println("chlayer_recursive, Parallel")
        o = iterate(node)
        while !isnothing(o)
            n, s = o
            chlayer_recursive(n)
            o = iterate(node, s)
        end
        if change_parallel
            chlayer(node)
        end
    end
    
    function chlayer_recursive(node)
        # println("chlayer_recursive, generic")
        # println("called on a $(typeof(node))")
        # println("looking for ", layers)
        if any(typeof(node) <: l for l ∈ layers)
            # println(typeof(node), " is in the list of layers to change")
            chlayer(node)
        end
    end
        
    chlayer_recursive(model)
end

true