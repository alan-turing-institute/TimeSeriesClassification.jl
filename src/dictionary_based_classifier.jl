using Statistics: std

function _discrete_fourier_transform(series, word_length, norm, inverse_sqrt_win_size, normalise)
    _length = length(series)
    output_length = ceil(Int, word_length/2)
    start = norm ? 1 : 0

    _std = 1
    
    if normalise
        s = std(series)  # some issue with std statement
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