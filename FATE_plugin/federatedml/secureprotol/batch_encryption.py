import datetime
import os
import shutil
import tempfile

import tensorflow as tf
import numpy as np
from numba import njit, prange
import math
import random
from federatedml.secureprotol.fate_paillier import PaillierPublicKey, PaillierPrivateKey
from federatedml.secureprotol import aciq

import multiprocessing
from joblib import Parallel, delayed, dump, load

import warnings
# import pysnooper

from arch.api.utils import log_utils
LOGGER = log_utils.getLogger()

N_JOBS = multiprocessing.cpu_count()


def encrypt(public_key: PaillierPublicKey, x):
    return public_key.encrypt(x)


def encrypt_array(public_key: PaillierPublicKey, A):
    # encrypt_A = []
    # for i in range(len(A)):
    #     encrypt_A.append(public_key.encrypt(float(A[i])))
    encrypt_A = Parallel(n_jobs=N_JOBS)(delayed(public_key.encrypt)(num) for num in A)
    return np.array(encrypt_A)


def encrypt_matrix(public_key: PaillierPublicKey, A):
    og_shape = A.shape
    if len(A.shape) == 1:
        A = np.expand_dims(A, axis=0)

    # print('encrypting matrix shaped ' + str(og_shape))
    A = np.reshape(A, (1, -1))
    A = np.squeeze(A)
    # print('max = ' + str(np.amax(A)))
    # print('min = ' + str(np.amin(A)))
    # encrypt_A = []
    # for i in range(len(A)):
    #     row = []
    #     for j in range(len(A[i])):
    #         if len(A.shape) == 3:
    #             row.append([public_key.encrypt(float(A[i, j, k])) for k in range(len(A[i][j]))])
    #         else:
    #             row.append(public_key.encrypt(float(A[i, j])))
    #
    #     encrypt_A.append(row)
    LOGGER.info("encrypt 1 matrix")
    encrypt_A = Parallel(n_jobs=N_JOBS)(delayed(public_key.encrypt)(num) for num in A)
    encrypt_A = np.expand_dims(encrypt_A, axis=0)
    encrypt_A = np.reshape(encrypt_A, og_shape)
    return np.array(encrypt_A)


@njit(parallel=True)
def stochastic_r(ori, frac, rand):
    result = np.zeros(len(ori), dtype=np.int32)
    for i in prange(len(ori)):
        if frac[i] >= 0:
            result[i] = np.floor(ori[i]) if frac[i] <= rand[i] else np.ceil(ori[i])
        else:
            result[i] = np.floor(ori[i]) if (-1 * frac[i]) > rand[i] else np.ceil(ori[i])
    return result


def stochastic_round(ori):
    rand = np.random.rand(len(ori))
    frac, decim = np.modf(ori)
    # result = np.zeros(len(ori))
    # for i in range(len(ori)):
    #     if frac[i] >= 0:
    #         result[i] = np.floor(ori[i]) if frac[i] <= rand[i] else np.ceil(ori[i])
    #     else:
    #         result[i] = np.floor(ori[i]) if (-1 * frac[i]) > rand[i] else np.ceil(ori[i])
    result = stochastic_r(ori, frac, rand)
    return result.astype(np.int)


def stochastic_round_matrix(ori):
    _shape = ori.shape
    ori = np.reshape(ori, (1, -1))
    ori = np.squeeze(ori)
    # rand = np.random.rand(len(ori))
    # frac, decim = np.modf(ori)

    # result = np.zeros(len(ori))
    #
    # for i in range(len(ori)):
    #     if frac[i] >= 0:
    #         result[i] = np.floor(ori[i]) if frac[i] <= rand[i] else np.ceil(ori[i])
    #     else:
    #         result[i] = np.floor(ori[i]) if (-1 * frac[i]) > rand[i] else np.ceil(ori[i])
    result = stochastic_round(ori)
    result = result.reshape(_shape)
    return result


def quantize_matrix(matrix, bit_width=8, r_max=0.5):
    og_sign = np.sign(matrix)
    uns_matrix = matrix * og_sign
    uns_result = (uns_matrix * (pow(2, bit_width - 1) - 1.0) / r_max)
    result = (og_sign * uns_result)
    # result = np.reshape(result, (1, -1))
    # result = np.squeeze(result)
    # # print(result)
    # result = stochastic_round(result)
    # print(result)
    return result, og_sign


def quantize_matrix_stochastic(matrix, bit_width=8, r_max=0.5):
    og_sign = np.sign(matrix)
    uns_matrix = matrix * og_sign
    uns_result = (uns_matrix * (pow(2, bit_width - 1) - 1.0) / r_max)
    result = (og_sign * uns_result)
    # result = np.reshape(result, (1, -1))
    # result = np.squeeze(result)
    # # print(result)
    result = stochastic_round_matrix(result)
    # print(result)
    return result, og_sign


def unquantize_matrix(matrix, bit_width=8, r_max=0.5):
    matrix = matrix.astype(int)
    og_sign = np.sign(matrix)
    uns_matrix = matrix * og_sign
    uns_result = uns_matrix * r_max / (pow(2, bit_width - 1) - 1.0)
    result = og_sign * uns_result
    return result.astype(np.float32)


def true_to_two_comp(input, bit_width):
    def true_to_two(value, bit_width):
        if value < 0:
            return 2 ** (bit_width + 1) + value
        else:
            return value

    # two_strings = [np.binary_repr(x, bit_width) for x in input]
    # # use 2 bits for sign
    # result = [int(x[0] + x, 2) for x in two_strings]
    result = Parallel(n_jobs=N_JOBS)(delayed(true_to_two)(x, bit_width) for x in input)
    return np.array(result)


@njit(parallel=True)
def true_to_two_comp_(input, bit_width):
    result = np.zeros(len(input), dtype=np.int32)
    for i in prange(len(input)):
        if input[i] >= 0:
            result[i] = input[i]
        else:
            result[i] = 2 ** (bit_width + 1) + input[i]
    return result


# @pysnooper.snoop('en_batch.log')
def encrypt_matrix_batch(public_key: PaillierPublicKey, A, batch_size=16, bit_width=8, pad_zero=3, r_max=0.5):
    og_shape = A.shape
    if len(A.shape) == 1:
        A = np.expand_dims(A, axis=0)

    # print('encrypting matrix shaped ' + str(og_shape) + ' ' + str(datetime.datetime.now().time()))
    A, og_sign = quantize_matrix(A, bit_width, r_max)

    A = np.reshape(A, (1, -1))
    A = np.squeeze(A)
    A = stochastic_round(A)

    # print("encrpting # " + str(len(A)) + " shape" + str(og_shape)+' ' + str(datetime.datetime.now().time()))

    A_len = len(A)
    # pad array at the end so tha the array is the size of
    A = A if (A_len % batch_size) == 0 \
        else np.pad(A, (0, batch_size - (A_len % batch_size)), 'constant', constant_values=(0, 0))

    # print('padded ' + str(datetime.datetime.now().time()))

    A = true_to_two_comp_(A, bit_width)

    # print([bin(x) for x in A])
    # print("encrpting padded # " + str(len(A))+' ' + str(datetime.datetime.now().time()))

    idx_range = int(len(A) / batch_size)
    idx_base = list(range(idx_range))
    # batched_nums = np.zeros(idx_range, dtype=int)
    batched_nums = np.array([pow(2, 2048)] * idx_range)
    batched_nums *= 0
    # print(batched_nums.dtype)
    for i in range(batch_size):
        idx_filter = [i + x * batch_size for x in idx_base]
        # print(idx_filter)
        filted_num = A[idx_filter]
        # print([bin(x) for x in filted_num])
        batched_nums = (batched_nums * pow(2, (bit_width + pad_zero))) + filted_num
        # print([bin(x) for x in batched_nums])

    # print("encrpting batched # " + str(len(batched_nums))+' ' + str(datetime.datetime.now().time()))

    # print([bin(x).zfill(batch_size*(bit_width+pad_zero) + 2) + ' ' for x in batched_nums])

    encrypt_A = Parallel(n_jobs=N_JOBS)(delayed(public_key.encrypt)(num) for num in batched_nums)

    # print('encryption done'+' ' + str(datetime.datetime.now().time()))

    return encrypt_A, og_shape


def encrypt_matmul(public_key: PaillierPublicKey, A, encrypted_B):
    """
     matrix multiplication between a plain matrix and an encrypted matrix

    :param public_key:
    :param A:
    :param encrypted_B:
    :return:
    """
    if A.shape[-1] != encrypted_B.shape[0]:
        print("A and encrypted_B shape are not consistent")
        exit(1)
    # TODO: need a efficient way to do this?
    res = [[public_key.encrypt(0) for _ in range(encrypted_B.shape[1])] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(encrypted_B.shape[1]):
            for m in range(len(A[i])):
                res[i][j] += A[i][m] * encrypted_B[m][j]
    return np.array(res)


def encrypt_matmul_3(public_key: PaillierPublicKey, A, encrypted_B):
    if A.shape[0] != encrypted_B.shape[0]:
        print("A and encrypted_B shape are not consistent")
        print(A.shape)
        print(encrypted_B.shape)
        exit(1)
    res = []
    for i in range(len(A)):
        res.append(encrypt_matmul(public_key, A[i], encrypted_B[i]))
    return np.array(res)


def decrypt(private_key: PaillierPrivateKey, x):
    return private_key.decrypt(x)


def decrypt_scalar(private_key: PaillierPrivateKey, x):
    return private_key.decrypt(x)


def decrypt_array(private_key: PaillierPrivateKey, X):
    decrypt_x = []
    for i in range(X.shape[0]):
        elem = private_key.decrypt(X[i])
        decrypt_x.append(elem)
    return decrypt_x


def encrypt_array(private_key: PaillierPrivateKey, X):
    decrpt_X = Parallel(n_jobs=N_JOBS)(delayed(private_key.decrypt())(num) for num in X)
    return np.array(decrpt_X)


def decrypt_matrix(private_key: PaillierPrivateKey, A):
    og_shape = A.shape
    if len(A.shape) == 1:
        A = np.expand_dims(A, axis=0)

    A = np.reshape(A, (1, -1))
    A = np.squeeze(A)

    # decrypt_A = []
    # for i in range(len(A)):
    #     row = []
    #     for j in range(len(A[i])):
    #         if len(A.shape) == 3:
    #             row.append([private_key.decrypt(A[i, j, k]) for k in range(len(A[i][j]))])
    #         else:
    #             row.append(private_key.decrypt(A[i, j]))
    #     decrypt_A.append(row)
    decrypt_A = Parallel(n_jobs=N_JOBS)(delayed(private_key.decrypt)(num) for num in A)
    decrypt_A = np.expand_dims(decrypt_A, axis=0)
    decrypt_A = np.reshape(decrypt_A, og_shape)
    return np.array(decrypt_A)



def two_comp_to_true(two_comp, bit_width=8, pad_zero=3):
    def binToInt(s, _bit_width=8):
        return int(s[1:], 2) - int(s[0]) * (1 << (_bit_width - 1))

    if two_comp < 0:
        raise Exception("Error: not expecting negtive value")
    two_com_string = bin(two_comp)[2:].zfill(bit_width + pad_zero)
    sign = two_com_string[0:pad_zero + 1]
    literal = two_com_string[pad_zero + 1:]

    if sign == '0' * (pad_zero + 1):  # positive value
        value = int(literal, 2)
        return value
    elif sign == '0' * (pad_zero - 2) + '1' + '0' * 2:  # positive value
        value = int(literal, 2)
        return value
    elif sign == '0' * pad_zero + '1':  # positive overflow
        value = pow(2, bit_width - 1) - 1
        return value
    elif sign == '0' * (pad_zero - 1) + '1' * 2:  # negtive value
        # if literal == '0' * (bit_width - 1):
        #     return 0
        return binToInt('1' + literal, bit_width)
    elif sign == '0' * (pad_zero - 2) + '1' * 3:  # negtive value
        # if literal == '0' * (bit_width - 1):
        #     return 0
        return binToInt('1' + literal, bit_width)
    elif sign == '0' * (pad_zero - 2) + '110':  # negtive overflow
        print('neg overflow: ' + two_com_string)
        return - (pow(2, bit_width - 1) - 1)
    else:  # unrecognized overflow
        print('unrecognized overflow: ' + two_com_string)
        # warnings.warn('Overflow detected, consider using longer r_max')
        return - (pow(2, bit_width - 1) - 1)


def two_comp_to_true_(two_comp, bit_width=8, pad_zero=3):
    def two_comp_lit_to_ori(lit, _bit_width):  # convert 2's complement coding of neg value to its original form
        return - 1 * (2 ** (_bit_width - 1) - lit)

    if two_comp < 0:
        raise Exception("Error: not expecting negtive value")
    # two_com_string = bin(two_comp)[2:].zfill(bit_width+pad_zero)

    sign = two_comp >> (bit_width - 1)
    literal = two_comp & (2 ** (bit_width - 1) - 1)

    if sign == 0:  # positive value
        return literal
    elif sign == 4:  # positive value, 0100
        return literal
    elif sign == 1:  # positive overflow, 0001
        return pow(2, bit_width - 1) - 1
    elif sign == 3:  # negtive value, 0011
        return two_comp_lit_to_ori(literal, bit_width)
    elif sign == 7:  # negtive value, 0111
        return two_comp_lit_to_ori(literal, bit_width)
    elif sign == 6:  # negtive overflow, 0110
        print('neg overflow: ' + str(two_comp))
        return - (pow(2, bit_width - 1) - 1)
    else:  # unrecognized overflow
        print('unrecognized overflow: ' + str(two_comp))
        warnings.warn('Overflow detected, consider using longer r_max')
        return - (pow(2, bit_width - 1) - 1)


def restore_shape(component, shape, batch_size=16, bit_width=8, pad_zero=3):
    num_ele = np.prod(shape)
    num_ele_w_pad = batch_size * len(component)
    # print("restoring shape " + str(shape))
    # print(" num_ele %d, num_ele_w_pad %d" % (num_ele, num_ele_w_pad))

    un_batched_nums = np.zeros(num_ele_w_pad, dtype=int)

    for i in range(batch_size):
        filter_ = (pow(2, bit_width + pad_zero) - 1) << ((bit_width + pad_zero) * i)
        # print(bin(filter))
        # filtered_nums = [x & filter for x in component]
        for j in range(len(component)):
            two_comp = (filter_ & component[j]) >> ((bit_width + pad_zero) * i)
            # print(bin(two_comp))
            un_batched_nums[batch_size * j + batch_size - 1 - i] = two_comp_to_true_(two_comp, bit_width, pad_zero)

    un_batched_nums = un_batched_nums[:num_ele]

    re = np.reshape(un_batched_nums, shape)
    # print("reshaped " + str(re.shape))
    return re


# @pysnooper.snoop('de_batch.log')
def decrypt_matrix_batch(private_key: PaillierPrivateKey, A, og_shape, batch_size=16, bit_width=8,
                         pad_zero=3, r_max=0.5):
    # A = [x.ciphertext(be_secure=False) if x.exponent == 0 else
    #      (x.decrease_exponent_to(0).ciphertext(be_secure=False) if x.exponent > 0 else
    #       x.increase_exponent_to(0).ciphertext(be_secure=False)) for x in A]

    # print("decrypting # " + str(len(A)) + " shape " + str(og_shape))

    decrypt_A = Parallel(n_jobs=N_JOBS)(delayed(private_key.decrypt)(num) for num in A)
    decrypt_A = np.array(decrypt_A)
    # print([bin(x).zfill(batch_size*(bit_width+pad_zero) + 2) for x in decrypt_A])

    result = restore_shape(decrypt_A, og_shape, batch_size, bit_width, pad_zero)

    result = unquantize_matrix(result, bit_width, r_max)
    return result


def calculate_clip_threshold(grads, theta=2.5):
    return [theta * np.std(x) for x in grads]


def calculate_clip_threshold_sparse(grads, theta=2.5):
    result = []
    for layer in grads:
        if isinstance(layer, tf.IndexedSlices):
            result.append(theta * np.std(layer.values.numpy()))
        else:
            result.append(theta * np.std(layer.numpy()))
    return result


def clip_with_threshold(grads, thresholds):
    return [np.clip(x, -1 * y, y) for x, y in zip(grads, thresholds)]


def clip_gradients_std(grads, std_theta=2.5):
    results = []
    thresholds = []
    for component in grads:
        clip_T = np.std(component) * std_theta
        thresholds.append(clip_T)
        results.append(np.clip(component, -1 * clip_T, clip_T))
    return results, thresholds


# def calculate_clip_threshold_aciq_g(grads, bit_width=8):
#     return [aciq.get_alpha_gaus(x, bit_width) for x in grads]


def calculate_clip_threshold_aciq_g(grads, grads_sizes, bit_width=8):
    res = []
    for idx in range(len(grads)):
        res.append(aciq.get_alpha_gaus(grads[idx], grads_sizes[idx], bit_width))
    # return [aciq.get_alpha_gaus(x, bit_width) for x in grads]
    return res


def calculate_clip_threshold_aciq_l(grads, bit_width=8):
    return [aciq.get_alpha_laplace(x, bit_width) for x in grads]


def batch_enc_per_layer(publickey, party, r_maxs, bit_width=16, batch_size=100, pad_zeros=3):
    result = []
    og_shapes = []

    for layer, r_max in zip(party, r_maxs):
        enc, shape_ = encrypt_matrix_batch(publickey, layer, batch_size=batch_size, bit_width=bit_width,
                                                      r_max=r_max, pad_zero=pad_zeros)
        # result.append(np.array(enc))
        result.append(enc)
        og_shapes.append(shape_)
    return result, og_shapes


def batch_dec_per_layer(privatekey, party, og_shapes, r_maxs, bit_width=16, batch_size=100, pad_zeros=3):
    result = []
    for layer, r_max, og_shape in zip(party, r_maxs, og_shapes):
        result.append(
            decrypt_matrix_batch(privatekey, layer, og_shape, batch_size=batch_size, bit_width=bit_width,
                                            r_max=r_max, pad_zero=pad_zeros).astype(np.float32))
    return np.array(result)


def sparse_to_dense(gradients):
    result = []
    for layer in gradients:
        if isinstance(layer, tf.IndexedSlices):
            result.append(tf.convert_to_tensor(layer).numpy())
        else:
            result.append(layer.numpy())
    return result


def aggregate_gradients_old(gradient_list, weight=0.5):
    def aggregate_layer(alllayers, weight=0.5):
        layer = np.add.reduce(alllayers, 0)
        return layer
    gradient_list = np.array(gradient_list)
    results = Parallel(n_jobs=len(gradient_list[0]), prefer="threads")(delayed(aggregate_layer)(gradient_list[:, i]) for i in range(len(gradient_list[0])))
    results = np.array(results)
    return results


# def aggregate_gradients_threading(gradient_list, weight=1):
#     def process_chunk(grad_flat, st, ed, w):
#         m_slice = grad_flat[:, st:ed]
#         return w * np.sum(m_slice, axis=0)
#
#     size = np.array([len(layer.flatten()) for layer in gradient_list[0]])
#     shape = [np.shape(layer) for layer in gradient_list[0]]
#     total = np.sum(size)
#     job_n = min(N_JOBS, total)
#     chunk_size = np.array([total // job_n] * job_n)
#     chunk_size[:total % job_n] += 1
#
#     grad_flat = []
#     for party in gradient_list:
#         host_grad_flat = [layer.flatten() for layer in party]
#         grad_flat.append(np.concatenate(host_grad_flat))
#     grad_flat = np.array(grad_flat)
#
#     ret = Parallel(n_jobs=job_n, backend='threading')(
#         delayed(process_chunk)(grad_flat, np.sum(chunk_size[:i]), np.sum(chunk_size[:i + 1]), weight)
#         for i in range(job_n)
#     )
#
#     split_index = [np.sum(size[:i + 1]) for i in range(len(size) - 1)]
#     result = np.split(np.concatenate(ret), split_index)
#     result = np.array([result[i].reshape(shape[i]) for i in range(len(shape))])
#     return result


# def aggregate_gradients_(gradient_list, weight=1):
#     def process_chunk(grad_flat, st, ed, w):
#         m_slice = grad_flat[:, st:ed]
#         return w * np.sum(m_slice, axis=0)
#
#     size = np.array([len(layer.flatten()) for layer in gradient_list[0]])
#     shape = [np.shape(layer) for layer in gradient_list[0]]
#     total = np.sum(size)
#     job_n = min(N_JOBS, total)
#     chunk_size = np.array([total // job_n] * job_n)
#     chunk_size[:total % job_n] += 1
#
#     grad_flat = []
#     for party in gradient_list:
#         host_grad_flat = [layer.flatten() for layer in party]
#         grad_flat.append(np.concatenate(host_grad_flat))
#     grad_flat = np.array(grad_flat)
#
#     temp_folder = tempfile.mkdtemp()
#     filename = os.path.join(temp_folder, 'joblib_test.mmap')
#     if os.path.exists(filename): os.unlink(filename)
#     _ = dump(grad_flat, filename)
#     grad_flat = load(filename, mmap_mode='r')
#
#     ret = Parallel(n_jobs=job_n, max_nbytes=None)(
#         delayed(process_chunk)(grad_flat, np.sum(chunk_size[:i]), np.sum(chunk_size[:i + 1]), weight)
#         for i in range(job_n)
#     )
#
#     try:
#         shutil.rmtree(temp_folder)
#     except OSError:
#         print('clean memmap failed')
#         pass
#
#     split_index = [np.sum(size[:i + 1]) for i in range(len(size) - 1)]
#     result = np.split(np.concatenate(ret), split_index)
#     result = np.array([result[i].reshape(shape[i]) for i in range(len(shape))])
#     return result


def aggregate_gradients(gradient_list, weight=1, mem_factor=1):
    def process_chunk(m_slice, w):
        # LOGGER.debug('start aggregating slice')
        re = np.sum(m_slice, axis=0)
        # LOGGER.debug('stop aggregating slice')
        return re

    gradient_list = [[np.array(layer) for layer in party] for party in gradient_list]

    size = np.array([len(layer.flatten()) for layer in gradient_list[0]])
    shape = [np.shape(layer) for layer in gradient_list[0]]
    total = np.sum(size)
    job_n = min(N_JOBS, total)
    # LOGGER.debug('aggregation job n ' + str(job_n))
    chunk_size = np.array([total // job_n] * job_n)
    chunk_size[:total % job_n] += 1

    grad_flat = []
    for party in gradient_list:
        host_grad_flat = [layer.flatten() for layer in party]
        grad_flat.append(np.concatenate(host_grad_flat))
    grad_flat = np.array(grad_flat)

    m_slices = []
    # temp_folder = tempfile.mkdtemp()
    for i in range(job_n):
        # filename = os.path.join(temp_folder, str(i) + 'joblib_test.mmap')
        st = np.sum(chunk_size[:i])
        ed = np.sum(chunk_size[:i + 1])
        m_slice = grad_flat[:, st:ed]
        # if os.path.exists(filename): os.unlink(filename)
        # _ = dump(m_slice, filename)
        # m_slice = load(filename, mmap_mode='r')
        m_slices.append(m_slice)

    del grad_flat

    ret = Parallel(n_jobs=job_n // mem_factor)(
        delayed(process_chunk)(m_slice, weight)
        for m_slice in m_slices
    )

    # try:
    #     shutil.rmtree(temp_folder)
    # except OSError:
    #     print('clean memmap failed')
    #     pass

    split_index = [np.sum(size[:i + 1]) for i in range(len(size) - 1)]
    result = np.split(np.concatenate(ret), split_index)
    del ret
    result = np.array([result[i].reshape(shape[i]) for i in range(len(shape))])
    return result


def calculate_clip_threshold_aciq_dis(layer_size, grads_mean, grads_mean_sqr, num_party, theta=2.5):
    clipping_thresholds = theta * (
            np.sum(grads_mean_sqr * layer_size, 0) / (layer_size * num_party)
            - (np.sum(grads_mean * layer_size, 0) / (layer_size * num_party)) ** 2) ** 0.5
    return clipping_thresholds
