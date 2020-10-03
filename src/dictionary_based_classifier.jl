using Statistics: std #We have some problem in the calculation of the std

function BOSSfit(X,  y, 
    threshold::Float64=0.92,
    n_parameter_samples::Int64=250,
    max_ensemble_size::Int64=500,
    min_window::Int64=10,
    word_lengths::Array=[16, 14, 12, 10, 8],
    weights,
    max_win_len_prop::Int64= 1,
    norm_options::Array=[true, false],
    #norm = false    use normalise instead 
    alphabet_size::Int64=4,
    save_words::Bool=true, # for BOSSIndividual and false for SAF
    random_state=nothing, 
    levels::Int64=1,
    normalise_dft::Bool=true,  
    #bigrams = false   As it is always false in BOSS 
    remove_repeat_words::Bool=false,
    level_weights::Array=[1, 2, 4, 16, 32, 64, 128]
               )
    n_classes = unique(y) #Take it directly from the interface,           
    n_instances, series_length = size(X)
    max_window_searches = series_length ÷ 4
    max_window = series_length * max_win_len_prop
    win_inc = (max_window - min_window) ÷ max_window_searches
    word_length = word_lengths[1] # Try to use one word for it 
    # Try making a struct with following
    classifiers = []
    classifiers_word_len = []
    classifiers_accuracy = []
    
    max_acc = -1
    min_max_acc = -1

    for i = 1:length(norm_options)
        for win_size = min_window:win_inc:(max_window + 1) 
            # window_size = win_size   norm = normalise  Try to replace it with one word 
            inverse_sqrt_win_size = 1 / sqrt(win_size)
            breakpoints = _mcb(X, win_size, alphabet_size, word_length, norm_options[i], inverse_sqrt_win_size, normalise_dft)
            bags, words_lists = transform(X, word_length, win_size, norm_options[i], inverse_sqrt_win_size,
                                                                 series_length, breakpoints, save_words, normalise_dft)
            
            best_classifier_for_win_size = temp = bags                                                       
            best_acc_for_win_size = -1
            best_word_len = word_length
            for j=1:length(word_lengths)
                if j > 1
                    bags = _shorten_bags(word_lengths[j], words_lists)  # Fix function 
                end
                # Make Array or somthing for accuracy/ may be change the variable 
                accuracy = _individual_train_acc(bags, y, train_size, best_acc_for_win_size)
                if accuracy >= best_acc_for_win_size
                    best_acc_for_win_size = accuracy
                    best_classifier_for_win_size = bags
                    best_word_len = word_lengths[j]  
                end
            end
            if some_logic
                #def _clean(self):
                #    self.transformer.words = None
                #    self.transformer.save_words = False
                #def _set_word_len(self, word_len):
                #    self.word_length = word_len
                #    self.transformer.word_length = word_len
                push!(classifiers, best_classifier_for_win_size)
                push!(classifiers_word_len, best_word_len)
                push!(classifiers_accuracy, best_acc_for_win_size)
                if best_acc_for_win_size > max_acc
                    
                end
                if length(classifiers) > max_ensemble_size

                end
            end
            # saving the class if acc > 92 of the best one _include_in_ensemble
        end
    end        
end

function _discrete_fourier_transform(series, word_length, norm, inverse_sqrt_win_size, normalise_dft)
    _length = length(series)
    output_length = ceil(Int, word_length/2)
    start = norm ? 1 : 0

    _std = 1
    
    if normalise_dft
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
    
    if normalise_dft
        dft = dft * (inverse_sqrt_win_size/ _std)    
    end
    return dft
end

# self._create_word(), self._add_to_pyramid(),  self.levels,  self._add_to_bag(), self.bigrams (are not used ), 
# self.save_words, self.words.append, self.window_size, self.word_length, 

function transform(X, word_length, window_size, norm, inverse_sqrt_win_size, series_length, breakpoints, save_words, normalise_dft)    # What is diff b\w transform and _shorten_bags
    # check_is_fitted()
    # X = check_X(X, enforce_univariate=True)
    # X = tabularize(X, return_array=True)
    # bags = pd.DataFrame()
    dim = []
    levels = 1
    norm = norm # norm used in _discrete_fourier_transform and _mft while calcuation
    remove_repeat_words = false
    words_list = zeros(Float64, size(X)[1], size(dfts)[1])
    for i = 1:size(X)[1]
        dfts = _mft(X[i, :], word_length, window_size, norm, inverse_sqrt_win_size, series_length, normalise_dft)
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
            repeat_word = levels > 1 ? (_add_to_pyramid(bag, words[window], last_word, (window - (repeat_words ÷2)), 
                          remove_repeat_words, series_length, level_weights, levels) : _add_to_bag(bag, words[window], last_word, remove_repeat_words))
            
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
        append!(dim, bag)
    end
    return dim, words_list
end

function _shorten_bags(word_len, words_lists)      # Check the logic changes w.r.t transform function
    dim = []
    w_instance, w_length = size(words_lists)
    for i = 1:w_instance
        bag = Dict()
        last_word = -1
        repeat_words = 0
        new_words = zeros(Int64, w_length)
        for j = 1:w_length
            new_word_len = shorten(16 - word_len)
            new_words[j] = (words_lists[i, j] >> (2*new_word_len))    # This might be wrong check corner cases
            repeat_word = levels > 1 ? (_add_to_pyramid(bag, new_words[j], last_word, (window - (repeat_words ÷2)), 
                          remove_repeat_words, series_length, level_weights, levels) : _add_to_bag(bag, new_words[j], last_word, remove_repeat_words))
            if repeat_word
                repeat_words += 1
            else
                last_word = new_words[j]
                repeat_words = 0
            end
        end
        append!(dim, bag)
    end
    return dim
end

function _mft(series, word_length, window_size, norm, inverse_sqrt_win_size, series_length, normalise_dft)  # Relation b\w transform and mft while runing main loop
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
                                               norm, inverse_sqrt_win_size, normalise_dft) #normalise =false
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

function _mcb(X, window_size, alphabet_size, word_length, norm, inverse_sqrt_win_size, normalise_dft)
    n_instances, series_length = size(X)
    num_windows_per_inst = ceil(series_length / window_size)
    dft = zeros(Float64, n_instances, num_windows_per_inst, word_length)
    @inbounds for k=1:n_instances
        @inbounds for i=1:num_windows_per_inst-1
            dft[k, i, :] = _discrete_fourier_transform(view(X, k, ((i-1)*window_size + 1):(i*window_size)), 
                                                word_length, norm, inverse_sqrt_win_size, normalise_dft)
        end
        dft[k, end, :] = _discrete_fourier_transform(view(X, k, (series_length-window_size + 1):series_length), 
                                                word_length, norm, inverse_sqrt_win_size, normalise_dft)
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

function _train_predict(train_num, bags)          
    best_dist = maxintfloat(Float64)
    nn = nothing                                             # Check on what type nn should be?
    for n =  1:length(bags)
        if n == train_num
            continue
        end
        dist = boss_distance(bags[train_num], bags[n], best_dist)
        if dist < best_dist
            best_dist = dist
            nn = class_vals[n]                        # Make class_vals to loop over BOSSIndividual
        end
    end
    return nn
end

function _individual_train_acc(bags, y, train_size, lowest_acc)
    correct = 0
    required_correct = floor(Int64, lowest_acc * train_size)
    for i = 1:train_size
        if correct + train_size - i < required_correct     # Check the boundary case i or i-1 ?
            return -1
        end
        c = _train_predict(i, bags)
        if c == y[i]
            correct += 1
        end
    end
    return correct / train_size
end

function boss_distance(first, second, best_dist=maxintfloat(Float64))
    dist = 0
    if typeof(first) <: Dict                   
        for (word, val_a) in first       
            val_b = second[word]         # check the cal again 
            dist += (val_a - val_b) * (val_a - val_b)
            if dist >= best_dist
                return maxintfloat(Float64)
            end
        end
    else
        dist = sum( [ first[n] == 0 ? 0 : ( (first[n] - second[n]) * (first[n] - second[n]) ) # Check this code again 
                                         for n=1:length(first) ] )
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

