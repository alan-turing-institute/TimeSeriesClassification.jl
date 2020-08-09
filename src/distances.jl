using  NearestNeighbors: Metric
import Distances: evaluate

struct dwt <: Metric 
    w::Int64
end

evaluate(dist::dwt, a, b) = dwt_distance(a, b, dist.w)

"""
   `dwt_distance(a, b, w)` is the basic dynamic time wraping function.
where `a` & `b` are the time series matrices and `w` is the percentage of 

"""
function dwt_distance(a, b, w)
    l_a, l_b = length(a), length(b)
    FloatMax = maxintfloat(Float64)

    if w <= 0
        band = max(l_b, l_a)
    else
        band = floor(Int, w*max(l_b, l_a))
    end

    M = zeros(l_a+1,l_b+1)
    M[1, 1:end] .= FloatMax
    M[1:end, 1] .= FloatMax
    for k=1:l_a 
        M[k, 2:end] =  (a .-  b[k]).^2
    end
    for i=2:l_a+1
        jstart = max(2, i-band)
        jstop = min(l_b+1, i+band+1)
        idx_inf_left = i-band-1
        
        if idx_inf_left >= 1
            M[i, idx_inf_left] = FloatMax
        end
        for j=jstart:jstop-1
            im = i-1
            jm = j-1
            M[i,j] = M[i,j] + min(min(M[im,j], M[i,jm]), M[im,jm])
        end
        if jstop <= l_b + 1
            M[i,jstop] = FloatMax
        end
    end
    return M[l_a,l_b]   
end





