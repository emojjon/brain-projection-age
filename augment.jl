import Augmentor

function getpipeline(augstr::String)
    
    augconf = Dict{Symbol, Vector{Float32}}()
    while true
        mo = match(r"^\s*([A-Z](?:\s+[0-9.]+)*)(?:\s|$)", augstr)
        if isnothing(mo)
            println("Exiting outer augmentation parsing loop, rest of configuration string is '$augstr'")
            break
        else
            grpstr = mo[1]
            println("Found parameter group '$grpstr'")
            k = Symbol(grpstr[1:1])
            grpstr = SubString(grpstr, 3)
            augstr = SubString(augstr, mo.offset + length(mo.match))
            params = Vector{Float32}()
            while true
                mo = match(r"^\s*([0-9.]+)(?:\s|$)", grpstr)
                if isnothing(mo)
                    println("Exiting inner augmentation parsing loop, rest of group configuration string is '$grpstr'")
                    break
                else
                    println("Found parameter '$(mo[1])'")
                    push!(params, parse(Float32, mo[1]))
                    grpstr = SubString(grpstr, mo.offset + length(mo.match))
                end
            end
            augconf[k] = params
        end
    end
    return getpipeline(augconf)
    
end

function getpipeline(augconf::Dict)
    ops = []
    if haskey(augconf, :S)
        ps = augconf[:S]
        if length(ps) == 3
            if ps[1] > ps[3]
                (ps[1], ps[3]) = (ps[3], ps[1])
            end
            sv = range(ps[1], step = ps[2], stop = ps[3])
        else
            println("Scale parameters not understood ('$(ps)'), using default range 1.01:0.05:1.2")
            sv = range(Float32(1.01), step = Float32(0.05), stop = Float32(1.2))
        end
        s₁ = [i for i ∈ sv, j ∈ sv]
        s₂ = [j for i ∈ sv, j ∈ sv]
        push!(ops, Augmentor.Scale(s₁[:], s₂[:]))
    end
    if haskey(augconf, :X)
        ps = augconf[:X]
        if length(ps) == 3
            if ps[1] > ps[3]
                (ps[1], ps[3]) = (ps[3], ps[1])
            end
            xv = range(ps[1], step = ps[2], stop = ps[3])
        elseif length(ps) == 2
            if ps[1] > ps[2]
                (ps[1], ps[2]) = (ps[2], ps[1])
            end
            xv = range(ps[1], step = Float32(1), stop = ps[2])
        elseif length(ps) == 1
            xv = range(-abs(ps[1]), step = Float32(1), stop = abs(ps[1]))
        else
            println("Parameters for shearing along X-axis not understood ('$(ps)'), using default range -5.0:5.0")
            xv = range(Float32(-5), step = Float32(1), stop = Float32(5))
        end
        push!(ops, Augmentor.ShearX(xv))
    end
    if haskey(augconf, :Y)
        ps = augconf[:Y]
        if length(ps) == 3
            if ps[1] > ps[3]
                (ps[1], ps[3]) = (ps[3], ps[1])
            end
            yv = range(ps[1], step = ps[2], stop = ps[3])
        elseif length(ps) == 2
            if ps[1] > ps[2]
                (ps[1], ps[2]) = (ps[2], ps[1])
            end
            yv = range(ps[1], step = Float32(1), stop = ps[2])
        elseif length(ps) == 1
            yv = range(-abs(ps[1]), step = Float32(1), stop = abs(ps[1]))
        else
            println("Parameters for shearing along Y-axis not understood ('$(ps)'), using default range -5.0:5.0")
            yv = range(Float32(-5), step = Float32(1), stop = Float32(5))
        end
        push!(ops, Augmentor.ShearY(yv))
    end
    if haskey(augconf, :R)
        ps = augconf[:R]
        if length(ps) == 3
            if ps[1] > ps[3]
                (ps[1], ps[3]) = (ps[3], ps[1])
            end
            rv = range(ps[1], step = ps[2], stop = ps[3])
        elseif length(ps) == 2
            if ps[1] > ps[2]
                (ps[1], ps[2]) = (ps[2], ps[1])
            end
            rv = range(ps[1], step = Float32(1), stop = ps[2])
        elseif length(ps) == 1
            rv = range(-abs(ps[1]), step = Float32(1), stop = abs(ps[1]))
        else
            println("Rotation parameters not understood ('$(ps)'), using default range -5.0:5.0")
            rv = range(Float32(-5), step = Float32(1), stop = Float32(5))
        end
        push!(ops, Augmentor.Rotate(rv))
    end
    if haskey(augconf, :E)
        ps = augconf[:E]
        if length(ps) < 2 || length(ps) > 4
            println("Parameters for elastic distortion not understood ('$(ps)'), using default: ElasticDistortion(4, 4, 0.1, 4)")
            ps = [4, 4, Float32(0.1), Float32(4)]
        end
        push!(ops, Augmentor.ElasticDistortion(round(Int32, ps[1]), round(Int32, ps[2]), ps[3:end]...))
    end
    if length(ops) == 0
        return Augmentor.NoOp(), Augmentor.NoOp()
    else
        cpl = reduce(|>, ops)
        return cpl |> Augmentor.CropSize(256, 256), cpl |> Augmentor.CropSize(208, 256)
    end
end
    
function get_augment_loader(XX, augfactor, pl1, pl2)
    println("Augmenting the training data")
    tic = time_ns()
    loader₁ = Flux.DataLoader(XX, batchsize = 32, shuffle = true) # Some sunny day I might parametrise the batchsize
    XXaug = Dict{Symbol, Array}(:d₁ => [], :d₂ => [], :d₃ => [], :l => [])
    dlen = 0
    idx = 0
    println("time\tsize")
    print("\r$(lpad(round(Int32, (time_ns() - tic) / 10 ^ 9), 5))\t$(lpad(idx, 6))                          ")
    flush(stdout)
    for vs ∈ (:d₁, :d₂, :d₃)
        tt = XX[vs]
        augset_size = [xx for xx ∈ size(tt)]
        dlen = augset_size[4]
        augset_size[4] *= augfactor
        augset = similar(tt, augset_size...)
        augset[:, :, :, 1:dlen] = tt
        XXaug[vs] = augset
    end
    XXaug[:l] = similar(XX[:l], (size(XX[:l]))[1:end-1]..., dlen * augfactor)
    XXaug[:l][:, 1:dlen] = XX[:l]
    tmpb = Dict{Symbol, Array}()
    channels = size(X_trans)[3]
    tmpb[:d₁] = similar(X_trans, size(X_trans)[1:2]..., channels * 32) #bs
    tmpb[:d₂] = similar(X_coron, size(X_coron)[1:2]..., channels * 32) #bs        
    tmpb[:d₃] = tmpb[:d₂]
    pl = Dict(:d₁ => pl1, :d₂ => pl2, :d₃ => pl2)
    idx = dlen + 1
    print("\r$(lpad(round(Int32, (time_ns() - tic) / 10 ^ 9), 5))\t$(lpad(idx, 6))                          ")
    flush(stdout)
    p_print = 100 * 32 / (dlen * augfactor)
    for pass ∈ 2:augfactor
        for b ∈ loader₁
            blen = size(b.l)[2]
            for vs ∈ (:d₁, :d₂, :d₃)
                augset = XXaug[vs]
                ready = false
                while ! ready
                    ready = true
                    try
                        if blen < 32 #bs
                            augset[:, :, :, idx:idx + blen - 1] = reshape(Augmentor.augmentbatch!(tmpb[vs][:,:,1:blen * channels], reshape(b[vs], (size(b[vs]))[1:2]..., :), pl[vs]), size(b[vs]))
                        else
                            augset[:, :, :, idx:idx + blen - 1] = reshape(Augmentor.augmentbatch!(tmpb[vs], reshape(b[vs], (size(b[vs]))[1:2]..., :), pl[vs]), size(b[vs]))
                        end
                    catch e
                        ready = false
                    end
                end
            end
            XXaug[:l][:, idx:idx + blen - 1] = b[:l]
            idx += blen
            if rand() < p_print
                print("\r$(lpad(round(Int32, (time_ns() - tic) / 10 ^ 9), 5))\t$(lpad(idx, 6))                          ")
                flush(stdout)
            end
        end
    end
    # We must convert our dict to named tuple to make it compatible with other loaders (project specific convention)
    return Flux.DataLoader((d₁ = XXaug[:d₁], d₂ = XXaug[:d₂], d₃ = XXaug[:d₃], l = XXaug[:l]), batchsize = 32, shuffle = true) #bs
end
