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

# self._create_word(), self._add_to_pyramid(),  self.levels,  self._add_to_bag(), self.bigrams (are not used ), 
# self.save_words, self.words.append, self.window_size, self.word_length, 

function transform(X, y, word_length, window_size, norm, inverse_sqrt_win_size, series_length, breakpoints, save_words)    # What is diff b\w transform and _shorten_bags
    # check_is_fitted()
    # X = check_X(X, enforce_univariate=True)
    # X = tabularize(X, return_array=True)
    # bags = pd.DataFrame()
    # dim = []
    levels = 1
    norm = norm # norm used in _discrete_fourier_transform and _mft while calcuation
    remove_repeat_words = false
    words_list = zeros(Float64, size(X)[1], size(dfts)[1])
    for i = 1:size(X)[1]
        dfts = _mft(X[i, :], word_length, window_size, norm, inverse_sqrt_win_size, series_length )
        bag = Dict() # {} This is how they use the ditionary in the python 
        last_word = -1
        repeat_words = 0
        words = zeros(Float64, size(dfts)[1])
        for window = 1:size(dfts)[1]  
            # word = _create_word(dfts[window, :], breakpoints, alphabet_size, word_length)
            @inbounds for i = 1:word_length
                @inbounds for bp = 1:alphabet_size
                    if dfts[window, :] <= breakpoints[i][bp]
                        words[window] = (words[window] << 2) | bp  
                        break
                    end
                end
            end
            # append!(words, word)
            # levels  
            repeat_word = levels > 1 ? (_add_to_pyramid(bag, words[window], last_word, window - 
                                   (repeat_words ÷2)) : _add_to_bag(bag, words[window], last_word))
            
            if repeat_word               # chek if order of end or if else got messed up
                repeat_words += 1
            else
                last_word = words[window]
                repeat_words = 0       
            end
        end
       # bigrams removed     
        if save_words
            words_list[i, :] =  words    #  self.words is equl to words_list 
        end
        dim.append!(bag)
    end
    bags[0] = dim
    return bags
end

function _shorten_bags(word_len)
    new_boss = BOSSIndividual(window_size, word_len,
                              alphabet_size, norm,
                              save_words=save_words,
                              random_state=random_state)
    new_boss.transformer = self.transformer
    sfa = self.transformer._shorten_bags(word_len)
    new_boss.transformed_data = [series.to_dict() for series in
                                 sfa.iloc[:, 0]]
    new_boss.class_vals = self.class_vals
    new_boss.num_classes = self.num_classes
    new_boss.classes_ = self.classes_
    new_boss.class_dictionary = self.class_dictionary 
    new_boss._is_fitted = True
    return new_boss
end

# self.words word.word
function _shorten_bags(self, word_len):      # To loop over different word lenghts
    new_bags = pd.DataFrame()
    dim = []
    
    for i in range(len(self.words)):
        bag = {}
        last_word = -1
        repeat_words = 0
        new_words = []
        for window, word in enumerate(self.words[i]):
            new_word = _BitWord(word=word.word)
            new_word.shorten(16 - word_len)
            repeat_word = (self._add_to_pyramid(bag, new_word, last_word,
                                                window -
                                                int(repeat_words/2))
                           if self.levels > 1 else
                           self._add_to_bag(bag, new_word, last_word))
            if repeat_word:
                repeat_words += 1
            else:
                last_word = new_word.word
                repeat_words = 0
    
            if self.bigrams:
                new_words.append(new_words)
    
                if window - self.window_size >= 0 and window > 0:
                    bigram = new_words[window - self.window_size] \
                        .create_bigram(word, self.word_length)
                    if self.levels > 1:
                        bigram = (bigram, 0)
                    bag[bigram] = bag.get(bigram, 0) + 1
    
        dim.append(pd.Series(bag))
    
    new_bags[0] = dim
    
    return new_bags
end

function _mft(series, word_length, window_size, norm, inverse_sqrt_win_size, series_length )  # Relation b\w transform and mft while runing main loop
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

#=
function _create_word(dft, word)                      # check out this function
    word = 0.0
    @inbounds for i = 1:word_length
        @inbounds for bp = 1:alphabet_size
            if dft[i] <= breakpoints[i][bp]
                word = (word << 2) | letter   
                break
            end
        end
    end
    return word
end
=#
# self.remove_repeat_words

function _add_to_bag(bag, word, last_word, remove_repeat_words)
    if remove_repeat_words && word == last_word         #word.word 
        return false
    end
    bag[word] = get(bag, word, 0) + 1   # check out this get fucntion 
    return true
end
# self.levels, self.remove_repeat_words, self.window_size,  self.series_length, self.level_weights

function _add_to_pyramid(bag, word, last_word, window_ind, remove_repeat_words, series_length, level_weights, levels)
    if remove_repeat_words && word == last_word  # check if python `and` and `&&` from the julia are same 
        return false
    end
    start = 0
    for i = 0:(levels-1)
        num_quadrants = 2^i                # Check out this function pow  
        quadrant_size = series_length / num_quadrants
        pos = window_ind + (window_size ÷ 2)
        quadrant = start + (pos / quadrant_size)
        bag[(word, quadrant)] = (get(bag, (word, quadrant), 0) + level_weights[i+1])
        start += num_quadrants
    end
    return true
end

function BOSSfit(X,  y, min_window, alphabet_size, word_lengths, inverse_sqrt_win_size,
                  norm_options, series_length, max_win_len_prop, max_ensemble_size)
    n_instances, series_length = size(X)
    threshold = 0.92
    n_parameter_samples=250
    max_ensemble_size=500
    min_window = 10
    word_lengths = [16, 14, 12, 10, 8]
    n_classes = #Take it directly from the interface
    weights = [] 
    max_win_len_prop = 1
    norm_options = [true, false]
    max_window_searches = series_length ÷ 4
    max_window = series_length * max_win_len_prop
    win_inc = (max_window - min_window) ÷ max_window_searches
    max_acc = -1
    min_max_acc = -1
    for normalise in norm_options
        for win_size = min_window:win_inc:(max_window + 1) 
            
            window_size = win_size    # Try to replace it with one word 
            #norm = false    use normalise instead 
            alphabet_size = 4
            save_words = true # for BOSSIndividual and false for SAF
            random_state = nothing 
            word_length = word_lengths[1] # Try to use one word for it 
            alphabet_size = 4
            levels = 1
            #bigrams = false   As it is always false in BOSS 
            remove_repeat_words = false 
            inverse_sqrt_win_size = 1 / sqrt(window_size)
            level_weights = [1, 2, 4, 16, 32, 64, 128]

            outputfrom_boss = _mcb(X, win_size, alphabet_size, word_length, norm, inverse_sqrt_win_size, normalise)
            outputfrom_transform = transform(outputfrom_boss...)
            for n, word_len in enumerate(word_lengths)
                outputfrom_shorten_bags = shorten_bags(word_len, outputfrom_transform)  # function 
                # Make Array or somthing for accuracy/ may be change the variable 
                accuracy = _individual_train_acc(outputfrom_shorten_bags, y, train_size, lowest_acc, transformed_data)
            end
            # saving the class if acc > 92 of the best one _include_in_ensemble
        end
    end        
end

function _train_predict(train_num, transformed_data)          
    best_dist = maxintfloat(Float64)
    nn = nothing                                             # Check on what type nn should be?
    for n =  1:length(transformed_data)
        if n == train_num
            continue
        end
        dist = boss_distance(transformed_data[train_num], transformed_data[n], best_dist)
        if dist < best_dist
            best_dist = dist
            nn = class_vals[n]                        # Make class_vals to loop over BOSSIndividual
        end
    end
    return nn
end

function _individual_train_acc(boss, y, train_size, lowest_acc, transformed_data)
    correct = 0
    required_correct = floor(Int64, lowest_acc * train_size)
    for i = 1:train_size
        if correct + train_size - i < required_correct     # Check the boundary case i or i-1 ?
            return -1
        end
        c = _train_predict(i, transformed_data)
        if c == y[i]
            correct += 1
        end
    end
    return correct / train_size
end

function boss_distance(first, second, best_dist=maxintfloat(Float64))
    dist = 0
    if typeof(first) <: Dict                   
        for (word, val_a) in first       # checkout how to loop over Dict items in julia
            val_b = second.get(word, 0)
            dist += (val_a - val_b) * (val_a - val_b)
            if dist >= best_dist
                return maxintfloat(Float64)
            end
        end
    else
        dist = sum( [ first[n] == 0 ? 0 : ( (first[n] - second[n]) * (first[n] - second[n]) )
                                         for n in range(len(first)) ] )
    end
    return dist
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

