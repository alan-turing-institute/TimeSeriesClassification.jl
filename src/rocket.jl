using Distributions: Uniform, Normal
using StatsBase: mean, sample, standardize, ZScoreTransform
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
        index = i + 1   # This is change, check for dilation > 1
        for j =1:(length_l)
            if index > 0 && index <= n_timepoints    # and in julia
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
    #n_columns = 1
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
These methods will be used for multivariate problems.
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