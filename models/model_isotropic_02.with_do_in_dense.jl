function make_my_model(args...; kwargs...)
    # Should return the model. This function is not responsible for loading it onto the gpu
    # This model ignores all arguments. They are only there for compatibility reasons.
    function σ(x)
        return Flux.leakyrelu(x, 0.1f0)
    end
    
    function pad(x)
        padding = zeros(eltype(x), (24, size(x)[2:end]...))
        return cat(padding, x, padding; dims = 1)
    end
    
    function pad(x::CUDA.CuArray)
        padding = CUDA.CuArray(zeros(eltype(x), (24, size(x)[2:end]...)))
        return cat(padding, x, padding; dims = 1)
    end
    
    transversal_chain = Flux.Chain(
        Flux.Conv((3,3), 2 => 4, σ; pad = 1), # 256 x 256
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

    coronal_chain = Flux.Chain(
        pad,
        transversal_chain
        )
    
    sagital_chain = Flux.Chain(
        pad,
        transversal_chain
        )
    
    parallel_part = Flux.Parallel((x...) -> cat(x..., dims=1),
        transversal_chain,
        coronal_chain,
        sagital_chain
        )

    return Flux.Chain(parallel_part,
        Flux.Dropout(0.5),
        Flux.Dense(768, 10, σ),
        Flux.Dropout(0.5),
        Flux.Dense(10, 1)
        )
end

# model = make_my_model() # The function should be called explicitly
