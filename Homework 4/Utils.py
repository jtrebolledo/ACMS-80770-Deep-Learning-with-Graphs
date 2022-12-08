import numpy as np
import matplotlib.pyplot as plt



def g_hat(l, K, d_values, a):
    
    g_hat = 0

    for i in range(K+1):
        if (-l) >= 0 and (-l) < a:
            g_hat += d_values[i]*(np.cos(2*np.pi*i*(l/a + 1/2)))
        else:
            g_hat += d_values[i]*(0)

    return g_hat


def g_hat_i_plus(l, a, j, R, K, d_values):
    int_var = np.log(l) - a*(j-1)/R

    g_hat_i_value = g_hat(int_var, K, d_values, a)

    return g_hat_i_value


def g_hat_j(l, a, j, R, d_values, J, K):

    if j == 1:
        term_1 = R*d_values[0]**2

        term_2 = 0

        for i in range(1, K+1):
            term_2 += d_values[i]**2

        term_3 = 0
        for i in range(2, J+1):
            term_3 += np.abs(g_hat_i_plus(l, a, i, R, K, d_values))**2

        g_hat_j_value = np.sqrt(term_1 + R/2*term_2 - term_3 + 1e-5)
        
    
    elif j >= 2 and j <= J:
        g_hat_j_value = g_hat_i_plus(l, a, j, R, K, d_values)

    
    return g_hat_j_value

