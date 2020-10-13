using Distributions: Uniform, Normal
using StatsBase: mean, sample, standardize
using Random
using LinearAlgebra
using MultivariateStats: ridge

function RocketFit(X::Array, y::AbstractArray, num_kernels::Int64=10000, normalise::Bool=true, random_state=nothing)
    n_timepoints, n_columns = size(X)  #n_timepoints should be max of sereis length in multidimentional series!
    _X = standardize(ZScoreTransform, X, dims=2)
    random_state = typeof(random_state) == Int64 ? random_state : nothing
    kernels = generate_kernels(n_timepoints, num_kernels, random_state)
    _X = apply_kernels(_X, kernels)
    Parameters = ridge(_X, y, 0)
    return Parameters, Kernels
end

function Predict_new(Xnew::Array, kernels, Parameters::Array)
    _Xnew = standardize(ZScoreTransform, Xnew, dims=2)
    _Xnew = apply_kernels(_Xnew, kernels)
    y_hat = _Xnew*Parameters[1:end-1] + Parameters[end]
    return y_hat
end 

function generate_kernels(n_timepoints::Int64, num_kernels::Int64, seed::Union{Bool, Int64})
    if seed != false
        Random.seed!(seed)
    end
    candidate_lengths = Int32[7, 9, 11]
    lengths = sample(candidate_lengths, num_kernels)
    weights = zeros(Float64, sum(lengths))

    biases = zeros(Float64, num_kernels)
    dilations = zeros(Int32, num_kernels)
    paddings = zeros(Int32, num_kernels)
    a1 = 1
    for i=1:num_kernels
        _length = lengths[i]
        _weights = rand(Normal(),  _length)
        b1 = a1 + _length
        weights[a1:b1-1] = _weights .- mean(_weights)
        biases[i] = rand(Uniform(-1, 1))
        dilation = floor(Int32, 2^( rand( Uniform(0, log2((n_timepoints-1)/(_length-1)) ))))
        dilations[i] = dilation
        padding = rand((0,1)) == 1 ? ( ((_length - 1) * dilation) ÷ 2) : 0      # check //
        paddings[i] = padding
        a1 = b1
    end
    return weights, lengths, biases, dilations, paddings
end

function apply_kernel(X::AbstractArray, weights::AbstractArray, length_l::Int32, 
                                         bias::Float64, dilation::Int32, padding::Int32) # change name if used as length above
    n_timepoints = length(X)
    output_length = (n_timepoints + (2 * padding)) - ((length_l - 1) * dilation)
    _ppv = 0
    _max = -maxintfloat(Float64)
    endl = (n_timepoints + padding) - ((length_l - 1) * dilation)

    for i = -padding:(endl-1)
        _sum = bias
        index = i
        for j =1:(length_l)
            if index > 0 && index < (n_timepoints-1)    # and in julia
                _sum = _sum + weights[j] * X[index]
            end
            index = index + dilation
        end
        if _sum > _max
            _max = _sum
        end
        if _sum > 0
            _ppv += 1
        end
    end
    return _ppv/output_length, _max
end

function apply_kernels(X::Array, kernels)
    weights, lengths, biases, dilations, paddings = kernels
    n_columns = 1
    n_instances, _ = size(X)
    num_kernels = length(lengths)
    _X = zeros(Float64, n_instances, (num_kernels * 2))  # 2 features per kernel
    for i = 1:n_instances   
        a1 = 1  # for weights
        a2 = 1  # for features
        for j =1:num_kernels
            b1 = a1 + lengths[j]
            b2 = a2 + 2
            _X[i, a2:b2-1] .= apply_kernel(X[i, :], weights[a1:b1-1], 
                                      lengths[j], biases[j], dilations[j], paddings[j]) #check the =

            a1 = b1
            a2 = b2
        end
    end    
    return _X
end

#=
function _generate_kernels(n_timepoints, num_kernels, n_columns, seed)
    if seed != false
        Random.seed!(seed)
    end
    
    candidate_lengths = Int32[7, 9, 11]
    lengths = sample(candidate_lengths, num_kernels)
    num_channel_indices = zeros(Int32, num_kernels)

    @inbounds for i=1:num_kernels
        limit = min(n_columns, lengths[i])
        num_channel_indices[i] = floor(Int32, 2^(rand(Uniform(0, log2(limit + 1)))))
    end

    channel_indices = zeros(Int32, floor(Int32, sum(num_channel_indices)))

    weights = zeros(Float64, floor(Int32, dot(lengths, num_channel_indices)))

    biases = zeros(Float64, num_kernels)
    dilations = zeros(Int32, num_kernels)
    paddings = zeros(Int32, num_kernels)

    a1 = 1  # for weights
    a2 = 1  # for channel_indices

    @inbounds for i=1:num_kernels

        _length = lengths[i]
        _num_channel_indices = num_channel_indices[i]

        _weights = rand(Normal(), _num_channel_indices * _length)

        b1 = a1 + (_num_channel_indices * _length)
        b2 = a2 + _num_channel_indices

        a3 = 1  # Check index
        @inbounds for _ = 1:_num_channel_indices
            b3 = a3 + _length
            _weights[a3:b3-1] = _weights[a3:b3-1] .- mean(_weights[a3:b3-1])
            a3 = b3
        end
        weights[a1:b1-1] = _weights

        channel_indices[a2:b2-1] = sample([0:(n_columns-1);], _num_channel_indices, replace=false)

        biases[i] = rand(Uniform(-1, 1))

        dilation = floor(Int32, 2^( rand( Uniform(0, log2((n_timepoints-1)/(_length-1)) ))))

        dilations[i] = dilation

        padding = rand((0,1)) == 1 ? ( ((_length - 1) * dilation) ÷ 2) : 0      # check //
        paddings[i] = padding

        a1 = b1
        a2 = b2
    end

    return (
        weights,
        lengths,
        biases,
        dilations,
        paddings,
        num_channel_indices,
        channel_indices,
       )
end

function _apply_kernel_univariate(X, weights, length_l, bias, dilation, padding) # change name if used as length above
    n_timepoints = length(X)

    output_length = (n_timepoints + (2 * padding)) - ((length_l - 1) * dilation)

    _ppv = 0
    _max = -maxintfloat(Float64)

    endl = (n_timepoints + padding) - ((length_l - 1) * dilation)

    @inbounds for i = -padding:(endl-1)

        _sum = bias

        index = i

        @inbounds for j =1:(length_l)

            if index > -1 && index < n_timepoints    # and in julia
                _sum = _sum + weights[j] * X[index]
            end
            index = index + dilation
        end
        if _sum > _max
            _max = _sum
        end
        if _sum > 0
            _ppv += 1
        end
    end
    return _ppv/output_length, _max
end


function _apply_kernel_multivariate(X, weights, length_l, bias, dilation, padding, num_channel_indices, channel_indices)
    n_timepoints = size(X)

    output_length = (n_timepoints + (2 * padding)) - ((length_l - 1) * dilation)

    _ppv = 0
    _max = -maxintfloat(Float64)

    endl = (n_timepoints + padding) - ((length_l - 1) * dilation)

    @inbounds for i = -padding:(endl-1)

        _sum = bias

        index = i

        @inbounds for j =1:(length_l)

            if index > -1 && index < n_timepoints # and in julia

                @inbounds for k = 1:num_channel_indices
                    _sum = _sum + weights[k, j] * X[channel_indices[k], index]
                end
            index = index + dilation
            end
        end 
        if _sum > _max
            _max = _sum
        end
        if _sum > 0
            _ppv += 1
        end
    end
    return _ppv / output_length, _max
end


function _apply_kernels(X, kernels)
    
    weights, lengths, biases, dilations, paddings, num_channel_indices, channel_indices = kernels
    n_columns = 1
    n_instances, _ = size(X)
    num_kernels = length(lengths)

    _X = zeros(Float64, n_instances, (num_kernels * 2))  # 2 features per kernel

    for i = 1:n_instances   # check waht is Prange!!

        a1 = 1  # for weights
        a2 = 1  # for channel_indices
        a3 = 1  # for features

        for j =1:num_kernels

            b1 = a1 + num_channel_indices[j] * lengths[j]
            b2 = a2 + num_channel_indices[j]
            b3 = a3 + 2

            if num_channel_indices[j] == 1

                _X[i, a3:b3-1] = _apply_kernel_univariate(
                    X[i, channel_indices[a2-1]],
                    weights[a1:b1-1],
                    lengths[j],
                    biases[j],
                    dilations[j],
                    paddings[j],
                )

            else

                _weights = reshape(weights[a1:b1-1], (num_channel_indices[j], lengths[j]))

                _X[i, a3:b3-1] = _apply_kernel_multivariate(
                    X[i, :],
                    _weights,
                    lengths[j],
                    biases[j],
                    dilations[j],
                    paddings[j],
                    num_channel_indices[j],
                    channel_indices[a2:b2-1]
                )
            end
            a1 = b1
            a2 = b2
            a3 = b3
        end
    end
    return _X
end
=#
#=
generate_kernels(15, 15, 4)
[[0.880101,  0.0871314,  0.686631   ,  0.711061 ,  0.47675   ,  0.02939 ,  0.507077  ,  0.189366 ,  0.872247,  0.273379 ,  0.911017 ,  0.596499 ,  0.598206  ,  0.249084 , 0.47217],
[0.435589,  0.098212 ,  0.254763   ,  0.302835 ,  0.310202  ,  0.443647,  0.433144  ,  0.468549 ,  0.83222 ,  0.552739 ,  0.594155 ,  0.417699 ,  0.665754  ,  0.128877 , 0.218403],
[0.935826,  0.378591 ,  0.6143     ,  0.517655 ,  0.42978   ,  0.848391,  0.513153  ,  0.529093 ,  0.503109,  0.274182 ,  0.842161 ,  0.202342 ,  0.0806719 ,  0.281881 , 0.566475],
[0.193258,  0.465889 ,  0.567566   ,  0.651156 ,  0.943689  ,  0.441694,  0.880676  ,  0.0667404,  0.435167,  0.671816 ,  0.778371 ,  0.655404 ,  0.550676  ,  0.159571 , 0.0208601],
[0.300428,  0.687278 ,  0.650008   ,  0.595494 ,  0.850667  ,  0.252594,  0.630805  ,  0.20122  ,  0.8158  ,  0.702978 ,  0.721408 ,  0.920646 ,  0.509343  ,  0.471785 , 0.513735],
[0.195544,  0.226161 ,  0.247839   ,  0.737217 ,  0.14977   ,  0.653269,  0.554925  ,  0.492101 ,  0.768723,  0.609894 ,  0.730879 ,  0.852111 ,  0.832708  ,  0.34111  , 0.674806],
[0.86578 ,  0.862769 ,  0.0656711  ,  0.0385844,  0.65245   ,  0.77028 ,  0.00567522,  0.836233 ,  0.225051,  0.0236561,  0.0677111,  0.234133 ,  0.00382134,  0.669392 , 0.894257],
[0.925464,  0.0175314,  0.781616   ,  0.64636  ,  0.990632  ,  0.65277 ,  0.928035  ,  0.276683 ,  0.587605,  0.592843 ,  0.335044 ,  0.100917 ,  0.0726753 ,  0.186799 , 0.435986],
[0.277557,  0.532061 ,  0.0448004  ,  0.582915 ,  0.743386  ,  0.288427,  0.398361  ,  0.169407 ,  0.177191,  0.0880408,  0.698665 ,  0.397409 ,  0.0646802 ,  0.754448 , 0.930136],
[0.380551,  0.480265 ,  0.703772   ,  0.209036 ,  0.118661  ,  0.839555,  0.901763  ,  0.26695  ,  0.554568,  0.92186  ,  0.816098 ,  0.955786 ,  0.402835  ,  0.943157 , 0.227009],
[0.103782,  0.213406 ,  0.326483   ,  0.214773 ,  0.832574  ,  0.717718,  0.864526  ,  0.965116 ,  0.723869,  0.23784  ,  0.174167 ,  0.752306 ,  0.0441352 ,  0.241657 , 0.194464],
[0.339557,  0.406484 ,  0.132795   ,  0.991552 ,  0.590005  ,  0.562917,  0.319092  ,  0.239256 ,  0.326745,  0.526035 ,  0.0776796,  0.0710774,  0.718191  ,  0.979984 , 0.945204],
[0.584707,  0.253499 ,  0.078567   ,  0.506861 ,  0.00670014,  0.582541,  0.86835   ,  0.591016 ,  0.834213,  0.852741 ,  0.714804 ,  0.106546 ,  0.377878  ,  0.461853 , 0.334955],
[0.324466,  0.225918 ,  0.765714   ,  0.686313 ,  0.980547  ,  0.692595,  0.0160522 ,  0.850262 ,  0.299598,  0.598781 ,  0.353672 ,  0.608926 ,  0.661193  ,  0.507749 , 0.558913],
[0.530486,  0.901775 ,  0.000810416,  0.602984 ,  0.317041  ,  0.237949,  0.663792  ,  0.783146 ,  0.981976,  0.0812254,  0.734902 ,  0.704842 ,  0.740334  ,  0.838503 , 0.916542]]
=#