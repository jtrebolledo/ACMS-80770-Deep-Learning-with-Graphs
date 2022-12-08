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


def plotting_filter(Kappa, d0, d1, R, J, lambda_max):

    # Dilation Factor
    a = R*np.log(lambda_max)/(J-R+1)


    Lambda = np.linspace(0,2, 500)


    n = Lambda.shape[0]

    fig = plt.figure(figsize = (10,10))

    for p in range(1, J+1):
        g_hat_j_lambda = np.zeros(n)

        j_value = p

        for i in range(n):
            g_hat_j_lambda[i] = g_hat_j(Lambda[i], a, j_value, R, [d0, d1], J, Kappa)

        plt.plot(Lambda, g_hat_j_lambda, label = "Spectral Filter " + str(p))
    plt.legend()

    plt.xlabel("$\lambda$", fontsize = 14)
    plt.show()




Kappa = 1
d0 = d1 = 0.5
R = 3
J = 8
lambda_max = 2

plotting_filter(Kappa, d0, d1, R, J, lambda_max)

