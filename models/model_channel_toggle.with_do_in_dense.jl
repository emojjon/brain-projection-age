function Base.sort(t::NTuple{N, T}; alg::Base.Sort.Algorithm=Base.Sort.defalg(Vector{T}()), lt=Base.Order.isless, by=Base.Order.identity, rev::Bool=false, order::Base.Order.Ordering=Base.Order.Forward) where {N, T}
    v = collect(t)
    return NTuple{N, T}(sort(v; alg = alg, lt = lt, by = by, rev = rev, order = order))
end

function make_my_model(args...; kwargs...)
    # Should return the model. This function is not responsible for loading it onto the gpu
    
    channels = nothing
    ir = iterate(args)
    if !isnothing(ir)
        println("Has positional arguments")
        first = ir[1]
    else
        println("Does not have positional arguments")
        first = nothing
    end
    if !isnothing(first)
        ir = iterate(first)
        if isnothing(ir)
            println("Doesn't look like a collection")
            elem = nothing
        else
            println("Looks like a collection")
            elem = ir[1]
        end
    else
        elem = nothing
    end
    if isa(elem, Integer)
        println("Is a set of channels")
        if isa(first, Integer)
            channels = args
        else
            channels = first
        end
    elseif haskey(kwargs, :channels)
        println("Has a channel keyword argument")
        ir = iterate(kwargs[:channels])
        if !isnothing(ir)
            println("Looks like a collection")
            elem = ir[1]
        else
            println("Doesn't look like a collection")
            elem = nothing
        end
        if isa(elem, Integer)
            println("Is a set of channels")
            channels = kwargs[:channels]
        end
    end
    if isnothing(channels)
        channels = 1:6
        println("Channel model called without specifying channels")
        println("args:")
        for (i,a) in enumerate(args)
            println("  $i: $a")
        end
        println("\nkwargs:")
        for (k,v) in pairs(kwargs)
            println("  $k:\t $v")
        end
        exit(1)
    end
    
    println("Channel model called with:")
    println("args:")
    for (i,a) in enumerate(args)
        println("  $i: $a")
    end
    println("\nkwargs:")
    for (k,v) in pairs(kwargs)
        println("  $k:\t $v")
    end
    if length(channels) == 1
        chstr = "only channel $(channels[1])"
    else
        chstr = "channels "*join(sort(channels), ", ", " and ")
    end
    println("Building model with $chstr")
    
    chainmask = [2i in channels || 2i - 1 in channels for i in 1:3]
    chainchannels = [[2i - 1 in channels, 2i in channels] for i in 1:3]
    chaincardinality = sum(chainmask)
    
    function σ(x)
        return Flux.leakyrelu(x, 0.1f0)
    end

    if chainmask[1]
        
        if sum(chainchannels[1]) == 1
            if chainchannels[1][1]
                trred = (x -> x[:,:,1:1,:],)
            else
                trred = (x -> x[:,:,2:2,:],)
            end
        else
            trred = ()
        end
                        
        transversal_chain = Flux.Chain(
            trred...,
            Flux.Conv((3,3), sum(chainchannels[1]) => 4, σ; pad = 1), # 256 x 256
            Flux.Conv((3,3), 4 => 4; stride = 2, pad = 1, bias = false),
            Flux.BatchNorm(4, σ; affine=true, momentum = 0.01f0),
            Flux.Conv((3,3), 4 => 8, σ; pad = 1), # 128 x 128
            Flux.Conv((3,3), 8 => 8; stride = 2, pad = 1, bias = false),
            Flux.BatchNorm(8, σ; affine=true, momentum = 0.01f0),
            Flux.Conv((3,3), 8 => 16, σ; pad = 1), # 64 x 64
            Flux.Conv((3,3), 16 => 16; stride = 2, pad = 1, bias = false),
            Flux.BatchNorm(16, σ; affine=true, momentum = 0.01f0),
            Flux.Conv((3,3), 16 => 32, σ; pad = 1), # 32 x 32
            Flux.Conv((3,3), 32 => 32; stride = 2, pad = 1, bias = false),
            Flux.BatchNorm(32, σ; affine=true, momentum = 0.01f0),
            Flux.Conv((3,3), 32 => 64, σ; pad = 1), # 16 x 16
            Flux.Conv((3,3), 64 => 64; stride = 2, pad = 1, bias = false),
            Flux.BatchNorm(64, σ; affine=true, momentum = 0.01f0),
            Flux.Conv((3,3), 64 => 128, σ; pad = 1), # 8 x 8
            Flux.Conv((3,3), 128 => 128; stride = 2, pad = 1, bias = false),
            Flux.BatchNorm(128, σ; affine=true, momentum = 0.01f0),
            Flux.Conv((4,4), 128 => 256, σ), # 4 x 4 -> 1 x 1
            x -> dropdims(x, dims = (1, 2))
            )
    else
        # Because the result of this chain will be the first element of a concatenation it is important to signal
        # where the result of said concatenation should reside, even if it is an empty tensor.
        function tr_nil_f(x)
            return Array{Float32}(undef, (0, size(x)[end]))
        end
        function tr_nil_f(x::CUDA.CuArray)
            return CUDA.CuArray(Array{Float32}(undef, (0, size(x)[end])))
        end
        transversal_chain = Flux.Chain( tr_nil_f )
    end
    
    if chainmask[2]
        
        if sum(chainchannels[2]) == 1
            if chainchannels[2][1]
                cored = (x -> x[:,:,1:1,:],)
            else
                cored = (x -> x[:,:,2:2,:],)
            end
        else
            cored = ()
        end
                        
        coronal_chain = Flux.Chain( # 208 x 256 , pad = (1, 0)
            cored...,
            Flux.Conv((3,3), sum(chainchannels[2]) => 4, σ, pad = (1, 0)),
            Flux.Conv((3,3), 4 => 4, stride = 2, pad = (1, 1), bias = false), 
            Flux.BatchNorm(4, σ, affine=true, momentum = 0.01f0),
            Flux.Conv((3,3), 4 => 8, σ, pad = (1, 0)),
            Flux.Conv((3,3), 8 => 8, stride = 2, pad = (1, 1), bias = false),
            Flux.BatchNorm(8, σ, affine=true, momentum = 0.01f0),
            Flux.Conv((3,3), 8 => 16, σ, pad = (1, 0)),
            Flux.Conv((3,3), 16 => 16, stride = 2, pad = (1, 1), bias = false),
            Flux.BatchNorm(16, σ, affine=true, momentum = 0.01f0),
            Flux.Conv((3,3), 16 => 32, σ, pad = (1, 0)),
            Flux.Conv((3,3), 32 => 32, stride = 2, pad = (1, 1), bias = false),
            Flux.BatchNorm(32, σ, affine=true, momentum = 0.01f0),
            Flux.Conv((3,3), 32 => 64, σ, pad = (1, 0)),
            Flux.Conv((3,3), 64 => 64, stride = 2, pad = (1, 1), bias = false),
            Flux.BatchNorm(64, σ, affine=true, momentum = 0.01f0),
            Flux.Conv((3,3), 64 => 128, σ),
            Flux.Conv((3,3), 128 => 128, stride = 2, pad = 1, bias = false),
            Flux.BatchNorm(128, σ, affine=true, momentum = 0.01f0),
            Flux.Conv((3,3), 128 => 256, σ),
            x -> dropdims(x, dims = (1, 2))
            )
    else
        coronal_chain = Flux.Chain(
            x -> Array{Float32}(undef, (0, size(x)[end]))
            )
    end        

    if chainmask[3]
                
        if sum(chainchannels[3]) == 1
            if chainchannels[3][1]
                sared = (x -> x[:,:,1:1,:],)
            else
                sared = (x -> x[:,:,2:2,:],)
            end
        else
            sared = ()
        end
                        
        sagital_chain = Flux.Chain(
            sared...,
            Flux.Conv((3,3), sum(chainchannels[3]) => 4, σ, pad = (1, 0)),
            Flux.Conv((3,3), 4 => 4, stride = 2, pad = (1, 1), bias = false), 
            Flux.BatchNorm(4, σ, affine=true, momentum = 0.01f0),
            Flux.Conv((3,3), 4 => 8, σ, pad = (1, 0)),
            Flux.Conv((3,3), 8 => 8, stride = 2, pad = (1, 1), bias = false),
            Flux.BatchNorm(8, σ, affine=true, momentum = 0.01f0),
            Flux.Conv((3,3), 8 => 16, σ, pad = (1, 0)),
            Flux.Conv((3,3), 16 => 16, stride = 2, pad = (1, 1), bias = false),
            Flux.BatchNorm(16, σ, affine=true, momentum = 0.01f0),
            Flux.Conv((3,3), 16 => 32, σ, pad = (1, 0)),
            Flux.Conv((3,3), 32 => 32, stride = 2, pad = (1, 1), bias = false),
            Flux.BatchNorm(32, σ, affine=true, momentum = 0.01f0),
            Flux.Conv((3,3), 32 => 64, σ, pad = (1, 0)),
            Flux.Conv((3,3), 64 => 64, stride = 2, pad = (1, 1), bias = false),
            Flux.BatchNorm(64, σ, affine=true, momentum = 0.01f0),
            Flux.Conv((3,3), 64 => 128, σ),
            Flux.Conv((3,3), 128 => 128, stride = 2, pad = 1, bias = false),
            Flux.BatchNorm(128, σ, affine=true, momentum = 0.01f0),
            Flux.Conv((3,3), 128 => 256, σ),
            x -> dropdims(x, dims = (1, 2))
            )
    else
        sagital_chain = Flux.Chain(
            x -> Array{Float32}(undef, (0, size(x)[end]))
            )
    end

    parallel_part = Flux.Parallel((x...) -> cat(x..., dims=1),
        transversal_chain,
        coronal_chain,
        sagital_chain
        )

    return Flux.Chain(parallel_part,
            Flux.Dropout(0.5),
        Flux.Dense(chaincardinality * 256, 10, σ),
            Flux.Dropout(0.5),
        Flux.Dense(10, 1)
        )
end

#model = make_my_model() # The function should be called explicitly
