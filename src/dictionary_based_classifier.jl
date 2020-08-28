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

# self.series_length  self.window_size, self.n_instances, self.alphabet_size,  self.word_length

function _mcb(X, window_size, alphabet_size, word_length)
    n_instances, series_length = size(X)
    num_windows_per_inst = ceil(series_length / window_size)
    dft = Array([_mcb_dft(X[i, :], num_windows_per_inst) for i=1:n_instances])

    total_num_windows = n_instances * num_windows_per_inst
    breakpoints = zeros((word_length, alphabet_size))
    for letter in range(word_length)                        # sortring algo, making a array and finding the aproproate ramge to place 
        column = sort(
            np.array([round(dft[inst][window][letter] * 100) / 100   # Round function 
                      for window in range(num_windows_per_inst) for inst in
                      range(n_instances)]))

        bin_index = 0
        target_bin_depth = total_num_windows / alphabet_size

        for bp in range(alphabet_size - 1)
            bin_index += target_bin_depth
            breakpoints[letter][bp] = column[int(bin_index)]  # Fix the int here!
        end
        breakpoints[letter][alphabet_size - 1] = sys.float_info.max   # check this out in the system

    return breakpoints, series_length
end

# self.window_size, self.window_size

function _mcb_dft(series, num_windows_per_inst, window_size)
    # Splits individual time series into windows and returns the DFT for
    # each
    series_length = length(series)
    split = [view(series, ((i-1)*window_size + 1):i*window_size) for i=1:num_windows_per_inst]  # Check the if indexing is correct!
    last_split = view(series, (series_length-window_size):series_length)
    split = [split..., last_split]
    return [_discrete_fourier_transform(row) for row in split]  # find the way to have two indiaces for he same loops @inoans may help
end

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
