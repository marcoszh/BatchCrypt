#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import time
import functools
import multiprocessing
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import contrib
from sklearn.metrics import accuracy_score

from arch.api.utils import log_utils
from federatedml.framework.gradients import Gradients
from federatedml.framework.homo.procedure import aggregator
from federatedml.framework.homo.procedure import paillier_cipher
from federatedml.linear_model.logistic_regression.homo_zcl_fmnist.homo_lr_base import HomoLRBase
from federatedml.linear_model.linear_model_weight import LinearModelWeights as LogisticRegressionWeights
from federatedml.model_selection import MiniBatch
from federatedml.optim.gradient.homo_lr_gradient import LogisticGradient, TaylorLogisticGradient
from federatedml.protobuf.generated import lr_model_param_pb2
from federatedml.secureprotol import PaillierEncrypt, batch_encryption
from federatedml.util import consts
from federatedml.util import fate_operator

tf.enable_eager_execution()
tfe = contrib.eager

LOGGER = log_utils.getLogger()

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
N_JOBS = multiprocessing.cpu_count()
LEARNING_RATE = 0.005

MODEL_JSON_DIR = CUR_DIR + '/fmnist_init.json'
MODEL_WEIGHT_DIR = CUR_DIR + '/fmnist_init.h5'


class HomoLRHost(HomoLRBase):
    def __init__(self):
        super(HomoLRHost, self).__init__()
        self.gradient_operator = None
        self.loss_history = []
        self.is_converged = False
        self.role = consts.HOST
        self.aggregator = aggregator.Host()
        self.model_weights = None
        self.cipher = paillier_cipher.Host()

        self.zcl_encrypt_operator = PaillierEncrypt()

    def _init_model(self, params):
        super()._init_model(params)
        self.cipher.register_paillier_cipher(self.transfer_variable)
        if params.encrypt_param.method in [consts.PAILLIER]:
            self.use_encrypt = True
            self.gradient_operator = TaylorLogisticGradient()
            self.re_encrypt_batches = params.re_encrypt_batches
        else:
            self.use_encrypt = False
            self.gradient_operator = LogisticGradient()

    def fit(self, data_instances, validate_data=None):
        LOGGER.debug("Start data count: {}".format(data_instances.count()))

        self._abnormal_detection(data_instances)
        self.init_schema(data_instances)
        validation_strategy = self.init_validation_strategy(data_instances, validate_data)

        pubkey = self.cipher.gen_paillier_pubkey(enable=self.use_encrypt, suffix=('fit',))
        if self.use_encrypt:
            self.cipher_operator.set_public_key(pubkey)

        self.model_weights = self._init_model_variables(data_instances)
        w = self.cipher_operator.encrypt_list(self.model_weights.unboxed)
        self.model_weights = LogisticRegressionWeights(w, self.model_weights.fit_intercept)

        LOGGER.debug("After init, model_weights: {}".format(self.model_weights.unboxed))

        mini_batch_obj = MiniBatch(data_inst=data_instances, batch_size=self.batch_size)

        total_batch_num = mini_batch_obj.batch_nums

        if self.use_encrypt:
            re_encrypt_times = total_batch_num // self.re_encrypt_batches + 1
            LOGGER.debug("re_encrypt_times is :{}, batch_size: {}, total_batch_num: {}, re_encrypt_batches: {}".format(
                re_encrypt_times, self.batch_size, total_batch_num, self.re_encrypt_batches))
            self.cipher.set_re_cipher_time(re_encrypt_times)

        total_data_num = data_instances.count()
        LOGGER.debug("Current data count: {}".format(total_data_num))

        model_weights = self.model_weights
        degree = 0

        self.__synchronize_encryption()
        self.zcl_idx, self.zcl_num_party = self.transfer_variable.num_party.get(idx=0, suffix=('train',))
        LOGGER.debug("party num:" + str(self.zcl_num_party))
        self.__init_model()

        self.train_loss_results = []
        self.train_accuracy_results = []
        self.test_loss_results = []
        self.test_accuracy_results = []

        for iter_num in range(self.max_iter):
            # mini-batch
            LOGGER.debug("In iter: {}".format(iter_num))
            # batch_data_generator = self.mini_batch_obj.mini_batch_data_generator()
            batch_num = 0
            total_loss = 0
            epoch_train_loss_avg = tfe.metrics.Mean()
            epoch_train_accuracy = tfe.metrics.Accuracy()

            for train_x, train_y in self.zcl_dataset:
                LOGGER.info("Staring batch {}".format(batch_num))
                start_t = time.time()
                loss_value, grads = self.__grad(self.zcl_model, train_x, train_y)
                loss_value = loss_value.numpy()
                grads = [x.numpy() for x in grads]
                LOGGER.info("Start encrypting")
                loss_value = batch_encryption.encrypt(self.zcl_encrypt_operator.get_public_key(), loss_value)
                grads = [batch_encryption.encrypt_matrix(self.zcl_encrypt_operator.get_public_key(), x) for x in grads]
                LOGGER.info("Finish encrypting")
                grads = Gradients(grads)
                self.transfer_variable.host_grad.remote(obj=grads.for_remote(), role=consts.ARBITER, idx=0, suffix=(iter_num, batch_num))
                LOGGER.info("Sent grads")
                self.transfer_variable.host_loss.remote(obj=loss_value, role=consts.ARBITER, idx=0, suffix=(iter_num, batch_num))
                LOGGER.info("Sent loss")

                sum_grads = self.transfer_variable.aggregated_grad.get(idx=0, suffix=(iter_num, batch_num))
                LOGGER.info("Got grads")
                sum_loss = self.transfer_variable.aggregated_loss.get(idx=0, suffix=(iter_num, batch_num))
                LOGGER.info("Got loss")

                sum_loss = batch_encryption.decrypt(self.zcl_encrypt_operator.get_privacy_key(), sum_loss)
                sum_grads = [
                    batch_encryption.decrypt_matrix(self.zcl_encrypt_operator.get_privacy_key(), x).astype(np.float32)
                    for x
                    in sum_grads.unboxed]
                LOGGER.info("Finish decrypting")

                # sum_grads = np.array(sum_grads) / self.zcl_num_party

                self.zcl_optimizer.apply_gradients(zip(sum_grads, self.zcl_model.trainable_variables),
                                                   self.zcl_global_step)

                elapsed_time = time.time() - start_t
                # epoch_train_loss_avg(loss_value)
                # epoch_train_accuracy(tf.argmax(self.zcl_model(train_x), axis=1, output_type=tf.int32),
                #                      train_y)
                self.train_loss_results.append(sum_loss)
                train_accuracy_v = accuracy_score(train_y,
                                                  tf.argmax(self.zcl_model(train_x), axis=1, output_type=tf.int32))
                self.train_accuracy_results.append(train_accuracy_v)
                test_loss_v = self.__loss(self.zcl_model, self.zcl_x_test, self.zcl_y_test)
                self.test_loss_results.append(test_loss_v)
                test_accuracy_v = accuracy_score(self.zcl_y_test,
                                                 tf.argmax(self.zcl_model(self.zcl_x_test), axis=1,
                                                           output_type=tf.int32))
                self.test_accuracy_results.append(test_accuracy_v)

                LOGGER.info(
                    "Epoch {:03d}, iteration {:03d}: train_loss: {:.3f}, train_accuracy: {:.3%}, test_loss: {:.3f}, "
                    "test_accuracy: {:.3%}, elapsed_time: {:.4f}".format(
                        iter_num,
                        batch_num,
                        sum_loss,
                        train_accuracy_v,
                        test_loss_v,
                        test_accuracy_v,
                        elapsed_time)
                )

                batch_num += 1

                if batch_num >= self.zcl_early_stop_batch:
                    return

            self.n_iter_ = iter_num

    def __synchronize_encryption(self, mode='train'):
        """
        Communicate with hosts. Specify whether use encryption or not and transfer the public keys.
        """
        pub_key = self.transfer_variable.paillier_pubkey.get(idx=0, suffix=(mode,))
        LOGGER.debug("Received pubkey")
        self.zcl_encrypt_operator.set_public_key(pub_key)
        pri_key = self.transfer_variable.paillier_prikey.get(idx=0, suffix=(mode,))
        LOGGER.debug("Received prikey")
        self.zcl_encrypt_operator.set_privacy_key(pri_key)

    def __init_model(self):
        # self.zcl_model = keras.Sequential([
        #     keras.layers.Flatten(input_shape=(28, 28)),
        #     keras.layers.Dense(128, activation=tf.nn.relu),
        #     keras.layers.Dense(10, activation=tf.nn.softmax)
        # ])
        #
        # LOGGER.info("Initialed model")
        json_file = open(MODEL_JSON_DIR, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        loaded_model.load_weights(MODEL_WEIGHT_DIR)
        self.zcl_model = loaded_model
        LOGGER.info("Initialed model")

        # The data, split between train and test sets:
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255.0
        x_test /= 255.0
        y_train = y_train.squeeze().astype(np.int32)
        y_test = y_test.squeeze().astype(np.int32)

        avg_length = int(len(x_train) / self.zcl_num_party)
        split_idx = [_ * avg_length for _ in range(1, self.zcl_num_party)]
        x_train = np.split(x_train, split_idx)[self.zcl_idx]
        y_train = np.split(y_train, split_idx)[self.zcl_idx]

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        BATCH_SIZE = 128
        SHUFFLE_BUFFER_SIZE = 1000
        train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE)
        self.zcl_dataset = train_dataset
        self.zcl_x_test = x_test
        self.zcl_y_test = y_test

        self.zcl_cce = tf.keras.losses.SparseCategoricalCrossentropy()
        self.zcl_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        self.zcl_global_step = tf.Variable(0)

    def __loss(self, model, x, y):
        y_ = model(x)
        return self.zcl_cce(y_true=y, y_pred=y_)

    def __grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.__loss(model, inputs, targets)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def __clip_gradients(self, grads, min_v, max_v):
        results = [tf.clip_by_value(t, min_v, max_v).numpy() for t in grads]
        return results

    def predict(self, data_instances):

        LOGGER.info(f'Start predict task')
        self._abnormal_detection(data_instances)
        self.init_schema(data_instances)
        suffix = ('predict',)
        pubkey = self.cipher.gen_paillier_pubkey(enable=self.use_encrypt, suffix=suffix)
        if self.use_encrypt:
            self.cipher_operator.set_public_key(pubkey)

        if self.use_encrypt:
            final_model = self.transfer_variable.aggregated_model.get(idx=0, suffix=suffix)
            model_weights = LogisticRegressionWeights(final_model.unboxed, self.fit_intercept)
            wx = self.compute_wx(data_instances, model_weights.coef_, model_weights.intercept_)
            self.transfer_variable.predict_wx.remote(wx, consts.ARBITER, 0, suffix=suffix)
            predict_result = self.transfer_variable.predict_result.get(idx=0, suffix=suffix)
            predict_result = predict_result.join(data_instances, lambda p, d: [d.label, p, None,
                                                                                     {"0": None, "1": None}])

        else:
            predict_wx = self.compute_wx(data_instances, self.model_weights.coef_, self.model_weights.intercept_)
            pred_table = self.classify(predict_wx, self.model_param.predict_param.threshold)
            predict_result = data_instances.mapValues(lambda x: x.label)
            predict_result = pred_table.join(predict_result, lambda x, y: [y, x[1], x[0],
                                                                           {"1": x[0], "0": 1 - x[0]}])
        return predict_result

    def _get_param(self):
        header = self.header

        weight_dict = {}
        intercept = 0
        if not self.use_encrypt:
            lr_vars = self.model_weights.coef_
            for idx, header_name in enumerate(header):
                coef_i = lr_vars[idx]
                weight_dict[header_name] = coef_i
            intercept = self.model_weights.intercept_

        param_protobuf_obj = lr_model_param_pb2.LRModelParam(iters=self.n_iter_,
                                                             loss_history=self.loss_history,
                                                             is_converged=self.is_converged,
                                                             weight=weight_dict,
                                                             intercept=intercept,
                                                             header=header)
        from google.protobuf import json_format
        json_result = json_format.MessageToJson(param_protobuf_obj)
        LOGGER.debug("json_result: {}".format(json_result))
        return param_protobuf_obj
