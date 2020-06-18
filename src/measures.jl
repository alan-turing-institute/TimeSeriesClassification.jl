function L1(a, b)
    _l = length(a)
    TT =  0
    for i=1:_l
        if a[i] == b[i]
            TT = TT + 1
        end
    end
    return (TT*100)/_l
end
