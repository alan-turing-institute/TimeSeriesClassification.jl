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

np.random.seed(5)
kk = generate_kernels(15, 15)
XX = np.random.random((15,15))
apply_kernels(XX, kk)

XX = [0.58291082 0.77026    0.59999317 0.96146749 0.37977839 0.49701035 0.33833865 0.74275074 0.30049587 0.79878627 0.08594978 0.23566364 0.43816889 0.12604585 0.20303198
 0.84154274 0.37243245 0.21868173 0.84613282 0.57537467 0.63371648 0.88892854 0.71395906 0.26511267 0.00970831 0.54946346 0.70752623 0.80569461 0.19065633 0.63421428
 0.93855293 0.25818638 0.29248046 0.42564401 0.5711786  0.99005555 0.58819961 0.8878994  0.65621128 0.47277887 0.30554853 0.35017958 0.77713429 0.17750905 0.13793536
 0.21304496 0.66891344 0.37140495 0.20650268 0.3307044  0.08162956 0.43653555 0.81121598 0.44579246 0.44731015 0.02752846 0.27897767 0.0751705  0.04773838 0.86358558
 0.71431759 0.8304377  0.1329058  0.36528646 0.91321754 0.43338789 0.92565181 0.96136808 0.00660811 0.43293059 0.58311263 0.34882757 0.37603202 0.35582158 0.23103288
 0.02603724 0.731184   0.99851577 0.03834193 0.85909822 0.54430493 0.01951996 0.06881641 0.52544948 0.10456021 0.29599743 0.2748344  0.97060571 0.46966648 0.28518106
 0.93824209 0.96672667 0.71939988 0.13687903 0.88989899 0.44126029 0.78612738 0.72148873 0.65316408 0.76697313 0.81183026 0.46166154 0.48462061 0.09738624 0.37086277
 0.88613429 0.22027203 0.74891512 0.80662967 0.19656783 0.76460265 0.65734626 0.97545921 0.38829315 0.75675    0.84395362 0.37907264 0.22368808 0.3682424  0.65282988
 0.25738735 0.85849893 0.29519148 0.76403467 0.32104374 0.50432241 0.87418372 0.48539667 0.62418851 0.34280804 0.4685735  0.06141402 0.51976649 0.44915414 0.29898286
 0.17287711 0.34066372 0.29485052 0.3625377  0.80875406 0.89559956 0.91503332 0.42155562 0.89844751 0.58707759 0.9966111  0.12426284 0.75683156 0.58826857 0.51456602
 0.75821559 0.83328172 0.83108783 0.34411894 0.38067706 0.59528513 0.04287195 0.81470528 0.5860231  0.44350472 0.63813487 0.95535158 0.80421115 0.65614969 0.20754234
 0.97173602 0.38711375 0.90523631 0.50105462 0.67240828 0.88740297 0.05838575 0.29474664 0.17769359 0.04656562 0.34180828 0.91540073 0.25271628 0.11303782 0.94184494
 0.49331099 0.47179102 0.68784676 0.81708483 0.22690731 0.20235285 0.42449878 0.84759295 0.9271214  0.19040336 0.68945921 0.44586868 0.87811028 0.2956111  0.27020385
 0.02193189 0.87135088 0.16605978 0.2785312  0.2463312  0.92205218 0.08162816 0.33665844 0.68043503 0.16556516 0.5776585  0.42465708 0.01314008 0.60660289 0.34691688
 0.2115269  0.97515003 0.83898221 0.44870931 0.32124904 0.01409788 0.72806984 0.26632355 0.51168995 0.62601907 0.75290599 0.15814617 0.09131661 0.77406715 0.05145793]

apply_kernels_ans = 
[0.466666667  1.86915069  0.714285714   1.56404946  0.466666667  0.644749239  0.2    0.81511455 0.333333333     0.182182149           0.2   0.391800731  0.266666667  0.260279791 0.666666667 1.40668578 1.0  2.6588065         0.6 1.16076305 0.0  -0.831747841 0.533333333 2.42694389 0.866666667 2.13242483 0.714285714  0.97215377 0.428571429  1.04198929
 0.666666667   2.7605416  0.428571429   1.17048616          0.4    1.1636767  0.2     1.1492548 0.333333333  0.000722869176   0.266666667   0.965390512          0.2  0.473190683 0.777777778 1.18343075 1.0 1.61479977 0.666666667  2.3053302 0.0  -0.590403203 0.666666667  1.9228075         0.8 1.97877259 0.714285714  1.29059237 0.142857143 0.157695276 
 0.466666667  2.49380658  0.714285714  0.667454033  0.466666667   1.37855049  0.4   0.500730237 0.333333333     0.207603449   0.333333333   0.704937334  0.333333333  0.507972768         1.0 1.72705702 1.0 1.92037785 0.666666667 2.02551711 0.2   0.373990297 0.733333333 1.59525635 0.866666667  2.0680034 0.571428571  1.29283699 0.428571429 0.544295866
 0.666666667  3.20101023  0.571428571   1.03278335  0.466666667   1.06228991  0.2   0.630871416 0.333333333     0.627532089   0.133333333    1.15763168  0.266666667  0.391364466 0.888888889  1.5503435 1.0 1.34876186 0.533333333 1.40246686 0.0 -0.0529354498 0.733333333 1.78481358         0.8 1.85289043 0.285714286  2.00849779 0.285714286 0.902233787 
 0.666666667  2.13769425  0.714285714   1.13289562  0.466666667    1.9158521  0.4   0.998864346         0.0   -0.0485586352   0.333333333   0.350269008  0.266666667  0.786788254 0.888888889 1.14198556 0.8 1.40468465 0.466666667  2.9189426 0.0  -0.109600439 0.666666667 1.78128214 0.733333333  2.6235522 0.428571429  2.18413789 0.571428571  1.29903864 
 0.533333333   2.9836892  0.428571429   1.45750256          0.4   1.43926618  0.4   0.516466306 0.666666667     0.960756044   0.333333333     1.1196118          0.2   0.76917819 0.888888889 1.05131567 1.0 2.47633094 0.666666667 2.43674043 0.0  -0.925993035         0.6 2.69302784         0.8 2.77034219 0.714285714  2.40433854 0.428571429 0.958647645 
 0.533333333  1.71039666  0.428571429   1.11807856  0.333333333   1.48710231  0.6    0.22319701         0.0    -0.358080617   0.266666667   0.403950255  0.133333333   0.36698616 0.888888889 1.33615988 1.0 1.57683181         0.4 1.91053745 0.2   0.272680734 0.533333333 2.83349575 0.866666667 1.64220126 0.571428571  2.04472092 0.428571429  0.68234057
         0.6  2.40716291  0.571428571  0.672292953          0.4  0.995636869  0.2    0.91441026 0.333333333      0.19570031   0.266666667   0.719213674          0.2   1.00029484         1.0 1.59276171 0.8 1.46403462 0.666666667 1.68903487 0.0  -0.257202796         0.8 2.15746543         0.8 2.33142911 0.285714286  1.58453787 0.285714286 0.276128956 
         0.6   2.5495622  0.428571429   1.04375563  0.533333333  0.907804279  0.0  -0.426021734 0.333333333     0.106019564   0.266666667   0.284415817          0.2  0.458375633 0.777777778 1.16355143 1.0 1.78442748 0.466666667 1.35029968 0.0  -0.296840664 0.666666667 1.94336411 0.933333333 1.87213404 0.714285714 0.710278128 0.142857143 0.184764197 
 0.533333333  1.92622824  0.285714286   1.26193378  0.533333333   1.21053493  0.2   0.330809842 0.333333333     0.182135788   0.266666667    0.51782547  0.133333333  0.091649982 0.777777778 1.53573269 0.6 2.48635508 0.666666667 1.45231436 0.4   0.692076558 0.666666667 2.04981721 0.866666667 2.50377807 0.285714286 0.514873952 0.285714286 0.604855995 
 0.466666667   2.8742218  0.571428571   1.12958146  0.466666667  0.982245782  0.2   0.502296872 0.333333333     0.529597288           0.4   0.343820151          0.2   0.43421471 0.888888889 1.35043108 0.8 1.35316112 0.533333333 1.95035689 0.2   0.353456767         0.6 2.82308158 0.866666667 2.12306767 0.714285714  2.49379564 0.142857143 0.439333312
 0.466666667  3.65026892  0.428571429   1.47874248  0.466666667   1.34404919  0.0  -0.140351928 0.333333333     0.859826084   0.333333333   0.803958376  0.333333333  0.569104927 0.777777778 1.57671415 1.0 2.60286486 0.533333333  2.0686166 0.0   -1.21365884 0.666666667 2.55552079         0.8 2.89945514 0.714285714  2.36488639 0.285714286  1.41428386 
 0.466666667  2.04937001  0.571428571   0.90878341  0.533333333   1.04658181  0.2   0.241451024 0.333333333     0.378097949           0.4   0.364750081  0.133333333  0.322329824 0.666666667 1.65218611 0.8 1.93793206 0.666666667 1.70518721 0.2 0.00126933835 0.666666667 2.20208141 0.933333333 2.08038651 0.571428571   1.4039941 0.285714286  0.57818107 
 0.666666667  2.27803077  0.571428571   1.43248392  0.666666667   1.01542443  0.2   0.989176771 0.333333333    0.0327688723   0.266666667   0.742917005  0.266666667  0.692247623 0.888888889 1.70813819 1.0 1.49204546 0.533333333 1.76710083 0.0  -0.247371683 0.933333333 1.63840312         0.8 2.21083423 0.285714286  2.72865965 0.428571429 0.938636132
 0.666666667  1.99437795  0.714285714  0.847532833  0.533333333  0.831622384  0.4   0.712211698 0.333333333      0.60095948           0.4    0.60247533          0.2  0.437287443 0.777777778 1.32221869 0.8 1.69595313 0.533333333 1.72321886 0.0  -0.354538113 0.666666667 2.81551915 0.933333333 2.23133105 0.428571429  2.02749729 0.285714286 0.862765838]

 weights   = [0.52762177, -0.0418873, 0.2471347, -1.31156877, -0.7247638, -0.78220244, -1.63412477, 2.7437687, -0.46801041, 0.01399379, 1.43003853, -0.68828345, 1.79152578, -0.19943179, -0.12082104, -0.18071698, -0.06975319, -0.47113781, 0.78166691, -0.84304843, 0.40557851, -1.27548192, 0.5544292, 0.44486959, -0.76179568, -0.84353163, -0.27708717, 0.05488085, 0.93928732, 0.81888716, -0.06003624, 1.42705634, -1.65799853, -1.03679063, -0.92473863, 0.80497937, -0.45438938, 1.91288138, 0.30939407, -0.22241857, -0.78292932, 0.62495391, -0.08524614, 0.18349484, -0.16950302, -0.60015582, -0.02692903, -0.56489894, 1.2632381, 0.74928421, 0.13981419, -0.25654669, 0.31853533, 0.04552576, -0.12017115, -1.26448862, -0.93308081, 1.32112779, -0.48612077, 0.68934027, -0.64064021, 0.42020047, 0.29536018, -0.919411, 0.64127106, -1.0641436, -0.31604487, -0.26644779, 1.58622318, -0.05704011, 0.36556408, -0.2481109, 0.60494258, 0.81612145, 0.83206467, -0.72643367, -0.13443512, -0.08208581, 0.09331525, -1.41730465, 1.06946835, -1.01957853, -0.03607454, 0.30114602, -2.20579273, 0.62166328, -0.1753795, 0.89379478, 1.76691169, -1.20234354, -1.28683233, -0.34219971, -1.06585416, -0.00949176, 0.89073309, 0.61441468, 0.35051293, 0.68524821, 0.13848917, 0.25151074, -0.22653086, -0.51363513, -1.01036993, -1.41547553, 0.32466544, 1.0371033, 1.47076476, 0.10694709, -0.75073608, 2.03104583, -0.51633515, 0.35914072, 0.74269627, -0.7995558, -1.0662558, 1.92427295, -0.67610275, -0.78531807, 0.48564705, -2.31143814, 0.02183875, 0.72847231, -0.58188375, 1.19451164, -0.85269623, 0.85391231, 0.82931092, -0.11102697, 1.39071293, -0.88898299, -0.52789062, 0.74821341, -1.44155276] 
 lengths   = Int32[11, 9, 11, 11, 7, 9, 7, 7, 11, 7, 11, 7, 7, 9, 9]
 biases    = [0.68675053, 0.092913, 0.13270275, -0.79834545, -0.10765975, -0.28003361, -0.33196979, 0.62634105, 0.87208684, 0.05435767, -0.80174301, 0.45357874, 0.68722719, 0.28728044, -0.46194353]
 dilations = Int32[1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1]
 paddings  = Int32[5, 0, 5, 0, 0, 4, 3, 0, 0, 6, 0, 3, 3, 0, 0]


 X = [0.58291082, 0.77026, 0.59999317, 0.96146749, 0.37977839, 0.49701035, 0.33833865, 0.74275074, 0.30049587, 0.79878627, 0.08594978, 0.23566364, 0.43816889, 0.12604585, 0.20303198]
W = [0.52762177, -0.0418873, 0.2471347, -1.31156877, -0.7247638, -0.78220244, -1.63412477, 2.7437687, -0.46801041, 0.01399379, 1.43003853]
bias = 0.68675053
l = Int32(11)
dial = Int32(1)
pad = Int32(5)
apply_kernel(X, W, l, bias, dial, pad) ≈ (0.4666666666666667, 1.8691506816260883)



 =#