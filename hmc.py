import numpy as np

def hmc(U, grad_U, eps, L, current_q, p_sampler):
    q = current_q
    p = p_sampler()
    current_p = p
    
    p -=  eps * grad_U(q)/2.0

    for i in range(L):
        q = q + eps *p
        if i != L-1:
            p -= eps * grad_U(q)

    p -= eps * grad_U(q)/2.0
    p = -p
    
    current_U = U(current_q)
    current_K = current_p.T.dot(current_p)*0.5
    proposed_U = U(q)
    proposed_K = p.T.dot(p)*0.5
    
    prob = np.exp(current_U - proposed_U + current_K - proposed_K)
    if np.random.uniform(size=1)< prob:
        return q, prob, 1
    else:
        return current_q, prob, 0
    
