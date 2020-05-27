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
from federatedml.linear_model.linear_model_weight import LinearModelWeights as LogisticRegressionWeights
from federatedml.linear_model.logistic_regression.homo_zcl_fmnist_batch.homo_lr_base import HomoLRBase
from federatedml.model_selection import MiniBatch
from federatedml.optim.gradient.homo_lr_gradient import LogisticGradient
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


class HomoLRGuest(HomoLRBase):
    def __init__(self):
        super(HomoLRGuest, self).__init__()
        self.gradient_operator = LogisticGradient()
        self.loss_history = []
        self.role = consts.GUEST
        self.aggregator = aggregator.Guest()

        self.zcl_encrypt_operator = PaillierEncrypt()

    def _init_model(self, params):
        super()._init_model(params)

    def fit(self, data_instances, validate_data=None):

        self._abnormal_detection(data_instances)
        self.init_schema(data_instances)

        validation_strategy = self.init_validation_strategy(data_instances, validate_data)
        self.model_weights = self._init_model_variables(data_instances)

        max_iter = self.max_iter
        total_data_num = data_instances.count()
        mini_batch_obj = MiniBatch(data_inst=data_instances, batch_size=self.batch_size)
        model_weights = self.model_weights

        self.__synchronize_encryption()
        self.zcl_idx, self.zcl_num_party = self.transfer_variable.num_party.get(idx=0, suffix=('train',))
        LOGGER.debug("party num:" + str(self.zcl_num_party))
        self.__init_model()

        self.train_loss_results = []
        self.train_accuracy_results = []
        self.test_loss_results = []
        self.test_accuracy_results = []

        batch_num = 0

        for iter_num in range(self.max_iter):
            total_loss = 0
            # batch_num = 0
            iter_num_ = 0
            epoch_train_loss_avg = tfe.metrics.Mean()
            epoch_train_accuracy = tfe.metrics.Accuracy()

            for train_x, train_y in self.zcl_dataset:
                LOGGER.info("Staring batch {}".format(batch_num))
                start_t = time.time()
                loss_value, grads = self.__grad(self.zcl_model, train_x, train_y)
                loss_value = loss_value.numpy()
                grads = [x.numpy() for x in grads]

                sizes = [layer.size * self.zcl_num_party for layer in grads]
                guest_max = [np.max(layer) for layer in grads]
                guest_min = [np.min(layer) for layer in grads]

                # clipping_thresholds_guest = batch_encryption.calculate_clip_threshold_aciq_l(grads, bit_width=self.bit_width)
                grad_max_all = self.transfer_variable.host_grad_max.get(idx=-1, suffix=(iter_num_, batch_num))
                grad_min_all = self.transfer_variable.host_grad_min.get(idx=-1, suffix=(iter_num_, batch_num))
                grad_max_all.append(guest_max)
                grad_min_all.append(guest_min)
                max_v = []
                min_v = []
                for layer_idx in range(len(grads)):
                    max_v.append([np.max([party[layer_idx] for party in grad_max_all])])
                    min_v.append([np.min([party[layer_idx] for party in grad_min_all])])
                grads_max_min = np.concatenate([np.array(max_v), np.array(min_v)], axis=1)
                clipping_thresholds = batch_encryption.calculate_clip_threshold_aciq_g(grads_max_min, sizes, bit_width=self.bit_width)
                LOGGER.info("clipping threshold " + str(clipping_thresholds))

                r_maxs = [x * self.zcl_num_party for x in clipping_thresholds]
                self.transfer_variable.clipping_threshold.remote(obj=clipping_thresholds, role=consts.HOST, idx=-1,
                                                                 suffix=(iter_num_, batch_num))
                grads = batch_encryption.clip_with_threshold(grads, clipping_thresholds)

                LOGGER.info("Start batch encrypting")
                loss_value = batch_encryption.encrypt(self.zcl_encrypt_operator.get_public_key(), loss_value)
                # grads = [batch_encryption.encrypt_matrix(self.zcl_encrypt_operator.get_public_key(), x) for x in grads]
                enc_grads, og_shape = batch_encryption.batch_enc_per_layer(
                    publickey=self.zcl_encrypt_operator.get_public_key(), party=grads, r_maxs=r_maxs,
                    bit_width=self.bit_width,
                    batch_size=self.e_batch_size)
                # grads = Gradients(enc_grads)
                LOGGER.info("Finish encrypting")
                # grads = self.encrypt_operator.get_public_key()
                # self.transfer_variable.guest_grad.remote(obj=grads.for_remote(), role=consts.ARBITER, idx=0,
                #                                          suffix=(iter_num_, batch_num))
                self.transfer_variable.guest_grad.remote(obj=enc_grads, role=consts.ARBITER, idx=0,
                                                         suffix=(iter_num_, batch_num))
                LOGGER.info("Sent grads")
                self.transfer_variable.guest_loss.remote(obj=loss_value, role=consts.ARBITER, idx=0,
                                                         suffix=(iter_num_, batch_num))
                LOGGER.info("Sent loss")

                sum_grads = self.transfer_variable.aggregated_grad.get(idx=0, suffix=(iter_num_, batch_num))
                LOGGER.info("Got grads")
                sum_loss = self.transfer_variable.aggregated_loss.get(idx=0, suffix=(iter_num_, batch_num))
                LOGGER.info("Got loss")

                sum_loss = batch_encryption.decrypt(self.zcl_encrypt_operator.get_privacy_key(), sum_loss)
                # sum_grads = [
                #     batch_encryption.decrypt_matrix(self.zcl_encrypt_operator.get_privacy_key(), x).astype(np.float32) for x
                #     in sum_grads.unboxed]
                sum_grads = batch_encryption.batch_dec_per_layer(privatekey=self.zcl_encrypt_operator.get_privacy_key(),
                                                                 # party=sum_grads.unboxed, og_shapes=og_shape,
                                                                 party=sum_grads, og_shapes=og_shape,
                                                                 r_maxs=r_maxs,
                                                                 bit_width=self.bit_width, batch_size=self.e_batch_size)
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

                # if batch_num >= self.zcl_early_stop_batch:
                #     return

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
        self._abnormal_detection(data_instances)
        self.init_schema(data_instances)
        predict_wx = self.compute_wx(data_instances, self.model_weights.coef_, self.model_weights.intercept_)

        pred_table = self.classify(predict_wx, self.model_param.predict_param.threshold)

        predict_result = data_instances.mapValues(lambda x: x.label)
        predict_result = pred_table.join(predict_result, lambda x, y: [y, x[1], x[0],
                                                                       {"1": x[0], "0": 1 - x[0]}])
        return predict_result
