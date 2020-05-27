import math
import numpy as np
# from scipy.special import erf
# import matplotlib.pyplot as plt
#
# plt.rcParams['figure.figsize'] = [16, 12]
import scipy.optimize as opt


def mse_laplace(alpha, b, num_bits):
    '''
    Calculating the sum of clipping error and quantization error for Laplace case

    Args:
    alpha: the clipping value
    b: location parameter of Laplace distribution
    num_bits: number of bits used for quantization

    Return:
    The sum of clipping error and quantization error
    '''
    return 2 * (b ** 2) * np.exp(-alpha / b) + (2 * alpha ** 2 * (2 ** num_bits - 2)) / (3 * (2 ** (3 * num_bits)))


def mse_gaussian(alpha, sigma, num_bits):
    '''
    Calculating the sum of clipping error and quantization error for Gaussian case

    Args:
    alpha: the clipping value
    sigma: scale parameter parameter of Gaussian distribution
    num_bits: number of bits used for quantization

    Return:
    The sum of clipping error and quantization error
    '''
    clipping_err = (sigma ** 2 + (alpha ** 2)) * (1 - math.erf(alpha / (sigma * np.sqrt(2.0)))) - \
                   np.sqrt(2.0 / np.pi) * alpha * sigma * (np.e ** ((-1) * (0.5 * (alpha ** 2)) / sigma ** 2))
    quant_err = (2 * alpha ** 2 * (2 ** num_bits - 2)) / (3 * (2 ** (3 * num_bits)))
    return clipping_err + quant_err


# To facilitate calculations, we avoid calculating MSEs from scratch each time.
# Rather, as N (0, sigma^2) = sigma * N (0, 1) and Laplace(0, b) = b * Laplace(0, 1),
# it is sufficient to store the optimal clipping values for N (0, 1) and Laplace(0, 1) and scale these
# values by sigma and b, which are estimated from the tensor values.

# Given b = 1, for laplace distribution
b = 1.
# print("Optimal alpha coeficients for laplace case, while num_bits falls in [2, 8].")
alphas = []
for m in range(2, 9, 1):
    alphas.append(opt.minimize_scalar(lambda x: mse_laplace(x, b=b, num_bits=m)).x)
# print(np.array(alphas))

# Given sigma = 1, for Gaussian distribution
sigma = 1.
# print("Optimal alpha coeficients for gaussian clipping, while num_bits falls in [2, 8]")
alphas = []
for m in range(2, 9, 1):
    alphas.append(opt.minimize_scalar(lambda x: mse_gaussian(x, sigma=sigma, num_bits=m)).x)
# print(np.array(alphas))


def get_alpha_laplace(values, num_bits):
    '''
    Calculating optimal alpha(clipping value) in Laplace case

    Args:
    values: input ndarray
    num_bits: number of bits used for quantization

    Return:
    Optimal clipping value
    '''

    # Dictionary that stores optimal clipping values for Laplace(0, 1)
    alpha_laplace = { 2: 2.83068299, 3: 3.5773953, 4: 4.56561968, 5: 5.6668432,
                    6: 6.83318852, 7: 8.04075143, 8: 9.27621011, 9: 10.53164388,
                    10: 11.80208734, 11: 13.08426947, 12: 14.37593053, 13: 15.67544068,
                    14: 16.98157905, 15: 18.29340105, 16: 19.61015778, 17: 20.93124164,
                    18: 22.25615278, 19: 23.58447327, 20: 24.91584992, 21:  26.24998231,
                    22: 27.58661098,  23: 28.92551169,   24: 30.26648869, 25: 31.60937055,
                    26: 32.9540057, 27: 34.30026003, 28: 35.64801378, 29: 36.99716035, 30: 38.3476039,
                    31: 39.69925781, 32: 41.05204406}

    # That's how ACIQ paper calcualte b
    b = np.mean(np.abs(values - np.mean(values)))
    return alpha_laplace[num_bits] * b


def get_alpha_gaus(values, num_bits):
    '''
    Calculating optimal alpha(clipping value) in Gaussian case

    Args:
    values: input ndarray
    num_bits: number of bits used for quantization

    Return:
    Optimal clipping value
    '''

    # Dictionary that stores optimal clipping values for N(0, 1)
    alpha_gaus = {2: 1.71063516, 3: 2.02612148, 4: 2.39851063, 5: 2.76873681,
                  6: 3.12262004, 7: 3.45733738, 8: 3.77355322, 9: 4.07294252,
                  10: 4.35732563, 11: 4.62841243, 12: 4.88765043, 13: 5.1363822,
                  14: 5.37557768, 15: 5.60671468, 16: 5.82964388, 17: 6.04501354, 18: 6.25385785,
                  19: 6.45657762, 20: 6.66251328, 21: 6.86053901, 22: 7.04555454, 23: 7.26136857,
                  24: 7.32861916, 25: 7.56127906, 26: 7.93151212, 27: 7.79833847, 28: 7.79833847,
                  29: 7.9253003, 30: 8.37438905, 31: 8.37438899, 32: 8.37438896}
    # That's how ACIQ paper calculate sigma, based on the range (efficient but not accurate)
    gaussian_const = (0.5 * 0.35) * (1 + (np.pi * np.log(4)) ** 0.5)
    sigma = ((np.max(values) - np.min(values)) * gaussian_const) / ((2 * np.log(values.size)) ** 0.5)
    return alpha_gaus[num_bits] * sigma


def get_alpha_gaus(values, values_size, num_bits):
    '''
    Calculating optimal alpha(clipping value) in Gaussian case

    Args:
    values: input ndarray
    num_bits: number of bits used for quantization

    Return:
    Optimal clipping value
    '''

    # Dictionary that stores optimal clipping values for N(0, 1)
    alpha_gaus = { 2: 1.71063516, 3: 2.02612148, 4: 2.39851063, 5: 2.76873681,
                6: 3.12262004, 7: 3.45733738, 8: 3.77355322, 9: 4.07294252,
                10: 4.35732563, 11: 4.62841243, 12: 4.88765043, 13: 5.1363822,
                14: 5.37557768, 15: 5.60671468, 16: 5.82964388, 17: 6.04501354, 18: 6.25385785,
                19: 6.45657762, 20: 6.66251328, 21: 6.86053901, 22: 7.04555454, 23: 7.26136857,
                24: 7.32861916, 25: 7.56127906, 26: 7.93151212, 27:7.79833847, 28: 7.79833847,
                29: 7.9253003, 30: 8.37438905, 31: 8.37438899, 32: 8.37438896}
    # That's how ACIQ paper calculate sigma, based on the range (efficient but not accurate)
    gaussian_const = (0.5 * 0.35) * (1 + (np.pi * np.log(4)) ** 0.5)
    # sigma = ((np.max(values) - np.min(values)) * gaussian_const) / ((2 * np.log(values.size)) ** 0.5)
    sigma = ((np.max(values) - np.min(values)) * gaussian_const) / ((2 * np.log(values_size)) ** 0.5)
    return alpha_gaus[num_bits] * sigma



if __name__ == '__main__':
    values = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.05, 0.01], [0.06, 0]], [[-0.05, -0.06], [-0.01, 0.03]]])
    print("----Test----")
    for num_bits in range(2, 17):
        print("num of bits == {}".format(num_bits))
        print("Laplace clipping value:  {}".format(get_alpha_laplace(values, num_bits)))
        print("Gaussian clipping value: {}".format(get_alpha_gaus(values, num_bits)))
        print("------")
