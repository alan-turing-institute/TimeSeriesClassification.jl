using Statistics: std #We have some problem in the calculation of the std

function _discrete_fourier_transform(series, word_length, norm, inverse_sqrt_win_size, normalise)
    _length = length(series)
    output_length = ceil(Int, word_length/2)
    start = norm ? 1 : 0

    _std = 1
    
    if normalise
        s = std(series)  # some issue with std statement ??
        if s!=0
            _std = s
        end
    end 

    dft = [ sum( [ [  series[n] * cos(2 * π * (n-1) * i / _length), 
                     -series[n] * sin(2 * π * (n-1) * i / _length) ] 
                                                   for n=1:_length ] ) # sum end
                                                   for i = start:(start + output_length-1) ] #end of dft

    dft = vcat(dft...)    
    
    if normalise
        dft = dft * (inverse_sqrt_win_size/ _std)    
    end
    return dft
end

# self._create_word(), self._add_to_pyramid(),  self.levels,  self._add_to_bag(), self.bigrams, 
# self.save_words, self.words.append, self.window_size, self.word_length, 

function transform(self, X, y=None)
    # check_is_fitted()
    # X = check_X(X, enforce_univariate=True)
    # X = tabularize(X, return_array=True)
    # bags = pd.DataFrame()
    # dim = []
    for i in range(X.shape[0])
        dfts = _mft(X[i, :])
        bag = Dict() # {} This is how they use the ditionary in the python 
        last_word = -1
        repeat_words = 0
        words = []
        for window in range(dfts.shape[0])  # use size(dfts, 1)
            word = _create_word(dfts[window])
            append!(words, word)
            # repeat_word = (_add_to_pyramid(bag, word, last_word,      # adding aproproate if eles statement
            #                                     window -
            #                                     Int(repeat_words/2))
            #                if levels > 1 else
            #                _add_to_bag(bag, word, last_word))
            if repeat_word
                repeat_words += 1
            else
                last_word = word.word
                repeat_words = 0
            end
            if bigrams
                if ((window - window_size) >= 0) && (window > 0)
                    # bigram = words[window - window_size]\  # Adding the biagram finding its library 
                    #    .create_bigram(word, word_length)
                    if levels > 1
                        bigram = (bigram, 0)
                    end
                    bag[bigram] = bag.get(bigram, 0) + 1
                end
            end        
        if save_words
            words.append(words)
        end
    end
        dim.append(pd.Series(bag))
    bags[0] = dim
    end
end

function _mft(series, word_length, window_size, norm, inverse_sqrt_win_size, series_length )
    start_offset = norm ? 2 : 0
    lengthl = (word_length + word_length) ÷ 2

    phis = [  [cos(2 * π * (-((i * 2) + start_offset) / 2) / window_size), 
               -sin(2 * π * (-((i * 2) + start_offset) / 2) / window_size) ] 
                                          for i = 0:((lengthl÷2)-1) ] #end of dft    
    phis = vcat(phis...)
    endl = max(1, series_length - window_size + 1)
    stds = _calc_incremental_mean_std(series, endl, window_size)   
    transformed = zeros(Float64, endl, lengthl)
    mft_data = []  # Find way to do catching 
    @inbounds for i = 0:endl-1
        if i > 0
            @inbounds for n = 1:2:lengthl
                real = mft_data[n] + series[i + window_size] - series[i]
                imag = mft_data[n + 1]
                mft_data[n] = real * phis[n] - imag * phis[n + 1]
                mft_data[n + 1] = real * phis[n + 1] + phis[n] * imag
            end
        else
            mft_data = _discrete_fourier_transform(series[1:window_size], word_length, 
                                               norm, inverse_sqrt_win_size, false) #normalise=false
        end
        normalising_factor = ( (1 / ( stds[i + 1] > 0 ? stds[i + 1] : 1) ) * inverse_sqrt_win_size) 
        transformed[i + 1, :] = mft_data * normalising_factor
    end
    return transformed
end

function _calc_incremental_mean_std(series, endl, window_size)
    means = zeros(endl)
    stds = zeros(endl)
    window = series[1:window_size]
    series_sum = sum(window)
    square_sum = sum(window.*window)
    
    r_window_length = 1 / window_size
    means[1] = series_sum * r_window_length
    buf = square_sum * r_window_length - means[1]^2
    stds[1] = buf > 0 ? sqrt(buf) : 0
    
    @inbounds for w =1:(endl-1)
        series_sum += series[w + window_size] - series[w]
        means[w + 1] = series_sum * r_window_length
        square_sum += series[w + window_size]^2 - series[w]^2
        buf = square_sum * r_window_length - means[w + 1]^2
        stds[w + 1] = buf > 0 ? sqrt(buf) : 0
    end
    return stds
end
# self.series_length  self.window_size, self.n_instances, self.alphabet_size,  self.word_length

function _mcb(X, window_size, alphabet_size, word_length, norm, inverse_sqrt_win_size, normalise)
    n_instances, series_length = size(X)
    num_windows_per_inst = ceil(series_length / window_size)
    dft = zeros(Float64, n_instances, num_windows_per_inst, word_length)
    @inbounds for k=1:n_instances
        @inbounds for i=1:num_windows_per_inst-1
            dft[k, i, :] = _discrete_fourier_transform(view(X, k, ((i-1)*window_size + 1):(i*window_size)), 
                                                word_length, norm, inverse_sqrt_win_size, normalise)
        end
        dft[k, end, :] = _discrete_fourier_transform(view(X, k, (series_length-window_size + 1):series_length), 
                                                word_length, norm, inverse_sqrt_win_size, normalise)
    end
    total_num_windows = n_instances * num_windows_per_inst
    breakpoints = zeros(Float64, word_length, alphabet_size)
    @inbounds for letter in range(word_length)                        # sortring algo, making a array and finding the aproproate ramge to place 
        sort!(dft[((letter-1)*total_num_windows + 1):(letter*total_num_windows)])
        bin_index = 0
        target_bin_depth = total_num_windows / alphabet_size
        @inbounds for bp in range(alphabet_size - 1)
            bin_index += target_bin_depth
            breakpoints[letter][bp] = dft[int(bin_index + (letter-1)*total_num_windows)]  # Check indexing here
        end
        breakpoints[letter][alphabet_size] = maxintfloat(Float64)   # check this out in the system
    return breakpoints
end

# self.window_size, self.window_size
#=
function _mcb_dft(series, num_windows_per_inst, window_size, series_length,
                        word_length, norm, inverse_sqrt_win_size, normalise)
    # Splits individual time series into windows and returns the DFT for
    # each
    Z = zeros(Float64, num_windows_per_inst, window_size)
    @inbounds for i=1:num_windows_per_inst-1
        Z[i, :] = _discrete_fourier_transform(series[((i-1)*window_size + 1):(i*window_size)], 
                                            word_length, norm, inverse_sqrt_win_size, normalise)
    end
    Z[end, :] = _discrete_fourier_transform(series[ (series_length-window_size + 1):series_length], 
                                            word_length, norm, inverse_sqrt_win_size, normalise)
    return Z # find the way to have two indiaces for he same loops @inoans may help
end
=#
# self.word_length, self.alphabet_size, self.breakpoints

function _create_word(self, dft)
    word = _BitWord()                       # check out this function
    for i in range(self.word_length)
        for bp in range(self.alphabet_size)
            if dft[i] <= self.breakpoints[i][bp]
                word.push(bp)
                break
            end
        end
    end
    return word
end

# self.remove_repeat_words

function _add_to_bag(self, bag, word, last_word)
    if self.remove_repeat_words && word.word == last_word
        return false
    end
    bag[word.word] = bag.get(word.word, 0) + 1   # check out this get fucntion 
    return true
end

# self.levels, self.remove_repeat_words, self.window_size,  self.series_length, self.level_weights

function _add_to_pyramid(self, bag, word, last_word, window_ind)
    if self.remove_repeat_words && word.word == last_word  # check if python `and` and `&&` from the julia are same 
        return false
    end
    start = 0
    for i in range(self.levels)
        num_quadrants = pow(2, i)                # Check out this fucntion 
        quadrant_size = self.series_length / num_quadrants
        pos = window_ind + Int((self.window_size / 2))
        quadrant = start + (pos / quadrant_size)
        bag[(word.word, quadrant)] = (bag.get((word.word, quadrant), 0)
                                      + self.level_weights[i])
        start += num_quadrants
    end
    return True
end

function BOSSfit(X,  y, min_window, alphabet_size, word_length, norm, inverse_sqrt_win_size,
                 normalise, series_length, max_win_len_prop, max_ensemble_size)

    max_window_searches = series_length ÷ 4
    max_window = series_length * max_win_len_prop
    win_inc = (max_window - min_window) ÷ max_window_searches
    inverse_sqrt_win_size = 1 / sqrt(window_size)
    for normalise in [true, false]
        for win_size = min_window:win_inc:(max_window + 1) 
            boss = _mcb(X, win_size, alphabet_size, word_length, norm, inverse_sqrt_win_size, normalise)
            _individual_train_acc(outputfrom_boss, y, )
end

function _train_predict(self, train_num)
    test_bag = self.transformed_data[train_num]
    best_dist = sys.float_info.max
    nn = None
    
    for n, bag in enumerate(self.transformed_data)
        if n == train_num
            continue
        end
        dist = boss_distance(test_bag, bag, best_dist)
        if dist < best_dist
            best_dist = dist
            nn = self.class_vals[n]
        end
    end
    return nn
end

function _individual_train_acc(self, boss, y, train_size, lowest_acc)
    correct = 0
    required_correct = int(lowest_acc * train_size)
    
    for i in range(train_size)
        if correct + train_size - i < required_correct
            return -1
        end
        c = boss._train_predict(i)
    
        if c == y[i]
            correct += 1
        end
    end
    return correct / train_size
end

"""
julia> linespace(2,4,5, Array{Int})
5-element Array{Int64,1}:
 2
 2
 3
 3
 4
"""
linespace(start,stop,_length, _type) =  convert(_type, floor.(LinRange(start,stop,_length)))
