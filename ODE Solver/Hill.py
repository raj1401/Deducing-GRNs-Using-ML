def hf(B,B_0,n,lmda):
    return lmda + (1.0 - lmda) * 1.0/(1+(B/B_0)**n)
    #return 1/(1+(B/B_0)**n) + lmda * B**n/(B_0**n+B**n)

def act_hf(prev_time,act_indices,param_set):
    prd = 1
    for i in range(len(act_indices)):
        ind = act_indices[i]
        if ind == 0:
            continue
        n = param_set[ind - 1]
        B_0 = param_set[ind - 2]
        lmda = param_set[ind]
        prd *= hf(prev_time[i],B_0,n,lmda) / lmda
    return prd

def inh_hf(prev_time,inh_indices,param_set):
    prd = 1
    for i in range(len(inh_indices)):
        ind = inh_indices[i]
        if ind == 0:
            continue
        n = param_set[ind - 1]
        B_0 = param_set[ind - 2]
        lmda = param_set[ind]
        prd *= hf(prev_time[i],B_0,n,lmda)
    return prd