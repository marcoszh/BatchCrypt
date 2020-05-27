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

import numpy as np
from functools import reduce

from arch.api.utils import log_utils
from federatedml.framework.gradients import Gradients
from federatedml.framework.homo.procedure import aggregator
from federatedml.framework.homo.procedure import paillier_cipher
from federatedml.linear_model.linear_model_weight import LinearModelWeights as LogisticRegressionWeights
from federatedml.linear_model.logistic_regression.homo_zcl_fmnist.homo_lr_base import HomoLRBase
from federatedml.optim import activation
from federatedml.secureprotol import PaillierEncrypt, batch_encryption
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HomoLRArbiter(HomoLRBase):
    def __init__(self):
        super(HomoLRArbiter, self).__init__()
        self.re_encrypt_times = []  # Record the times needed for each host

        self.loss_history = []
        self.is_converged = False
        self.role = consts.ARBITER
        self.aggregator = aggregator.Arbiter()
        self.model_weights = None
        self.cipher = paillier_cipher.Arbiter()
        self.host_predict_results = []

    def _init_model(self, params):
        super()._init_model(params)
        self.cipher.register_paillier_cipher(self.transfer_variable)

    def fit(self, data_instances=None, validate_data=None):
        host_ciphers = self.cipher.paillier_keygen(key_length=self.model_param.encrypt_param.key_length,
                                                   suffix=('fit',))
        num_host = len(host_ciphers)
        host_has_no_cipher_ids = [idx for idx, cipher in host_ciphers.items() if cipher is None]
        self.re_encrypt_times = self.cipher.set_re_cipher_time(host_ciphers)
        max_iter = self.max_iter
        validation_strategy = self.init_validation_strategy()

        self.__synchronize_encryption()

        self.transfer_variable.num_party.remote((0, num_host + 1), role=consts.GUEST, idx=0, suffix=('train',))
        # self.transfer_variable.num_party.remote(num_host + 1, role=consts.HOST, idx=-1, suffix=('train',))
        for _idx, _cipher in enumerate(host_ciphers):
            self.transfer_variable.num_party.remote((_idx + 1, num_host + 1), role=consts.HOST, idx=_idx,
                                                    suffix=('train',))

        for iter_num in range(self.max_iter):
            # re_encrypt host models
            # self.__re_encrypt(iter_num)
            batch_num = 0
            while batch_num >= 0:
                LOGGER.info("Staring batch {}".format(batch_num))
                LOGGER.info("Collecting grads & loss")
                guest_grad = self.transfer_variable.guest_grad.get(idx=0, suffix=(iter_num, batch_num))
                guest_grad = np.array(guest_grad.unboxed)
                LOGGER.debug(guest_grad.shape)
                guest_loss = self.transfer_variable.guest_loss.get(idx=0, suffix=(iter_num, batch_num))

                # guest_model = np.array(guest_model)
                LOGGER.info("received guest grads & loss")

                host_grads = self.transfer_variable.host_grad.get(idx=-1, suffix=(iter_num, batch_num))
                host_grads = [np.array(x.unboxed) for x in host_grads]
                [LOGGER.debug(x.shape) for x in host_grads]
                host_losses = self.transfer_variable.host_loss.get(idx=-1, suffix=(iter_num, batch_num))
                LOGGER.info("received host grads & loss")
                host_grads.append(guest_grad)
                # LOGGER.debug(host_grads.shape)
                host_losses.append(guest_loss)
                # sum_grads = self.__aggregate_grads_(host_grads)
                sum_grads = batch_encryption.aggregate_gradients(host_grads)
                sum_loss = self.__aggregate_losses(host_losses)

                LOGGER.info("Grads and loss aggregated")
                sum_grads = Gradients(sum_grads)
                self.transfer_variable.aggregated_grad.remote(obj=sum_grads.for_remote(), role=consts.GUEST, idx=0, suffix=(iter_num, batch_num))
                self.transfer_variable.aggregated_grad.remote(obj=sum_grads.for_remote(), role=consts.HOST, idx=-1, suffix=(iter_num, batch_num))
                LOGGER.info("Dispatched all grads")
                self.transfer_variable.aggregated_loss.remote(obj=sum_loss, role=consts.GUEST, idx=0,
                                                              suffix=(iter_num, batch_num))
                self.transfer_variable.aggregated_loss.remote(obj=sum_loss, role=consts.HOST, idx=-1,
                                                              suffix=(iter_num, batch_num))
                LOGGER.info("Dispatched all losses")

                batch_num += 1

                if batch_num >= self.zcl_early_stop_batch:
                    return


    def __synchronize_encryption(self, mode='train'):
        """
        Communicate with hosts. Specify whether use encryption or not and transfer the public keys.
        """
        # 2. Send pubkey to those use-encryption guest & hosts
        encrypter = PaillierEncrypt()
        encrypter.generate_key(self.key_length)

        pub_key = encrypter.get_public_key()

        # LOGGER.debug("Start to remote pub_key: {}, transfer_id: {}".format(pub_key, pubkey_id))
        self.transfer_variable.paillier_pubkey.remote(obj=pub_key, role=consts.GUEST, idx=0, suffix=(mode,))
        LOGGER.info("send pubkey to guest")
        pri_key = encrypter.get_privacy_key()
        self.transfer_variable.paillier_prikey.remote(obj=pri_key, role=consts.GUEST, idx=0, suffix=(mode,))
        # LOGGER.debug("Start to remote pri_key: {}, transfer_id: {}".format(pri_key, prikey_id))
        LOGGER.info("send prikey to guest")
        self.transfer_variable.paillier_pubkey.remote(obj=pub_key, role=consts.HOST, idx=-1, suffix=(mode,))
        LOGGER.info("send pubkey to host")
        self.transfer_variable.paillier_prikey.remote(obj=pri_key, role=consts.HOST, idx=-1, suffix=(mode,))
        LOGGER.info("send prikey to host")

    def __aggregate_grads(self, grad_list):
        def do_sum(x1, x2):
            results = []
            for i in range(len(x1)):
                results.append(x1[i] + x2[i])
            return results
        results_ = reduce(do_sum, grad_list)
        return results_

    def __aggregate_grads_(self, gradient_list, weight=0.5):
        results = np.add.reduce(gradient_list)
        return results

    def __aggregate_losses(self, loss_list):
        return np.sum(loss_list)

    def predict(self, data_instantces=None):
        LOGGER.info(f'Start predict task')
        current_suffix = ('predict',)
        host_ciphers = self.cipher.paillier_keygen(key_length=self.model_param.encrypt_param.key_length,
                                                   suffix=current_suffix)

        LOGGER.debug("Loaded arbiter model: {}".format(self.model_weights.unboxed))
        for idx, cipher in host_ciphers.items():
            if cipher is None:
                continue
            encrypted_model_weights = self.model_weights.encrypted(cipher, inplace=False)
            self.transfer_variable.aggregated_model.remote(obj=encrypted_model_weights.for_remote(),
                                                           role=consts.HOST,
                                                           idx=idx,
                                                           suffix=current_suffix)

        # Receive wx results

        for idx, cipher in host_ciphers.items():
            if cipher is None:
                continue
            encrypted_predict_wx = self.transfer_variable.predict_wx.get(idx=idx, suffix=current_suffix)
            predict_wx = cipher.distribute_decrypt(encrypted_predict_wx)

            prob_table = predict_wx.mapValues(lambda x: activation.sigmoid(x))
            predict_table = prob_table.mapValues(lambda x: 1 if x > self.model_param.predict_param.threshold else 0)

            self.transfer_variable.predict_result.remote(predict_table,
                                                         role=consts.HOST,
                                                         idx=idx,
                                                         suffix=current_suffix)
            self.host_predict_results.append((prob_table, predict_table))

    def run(self, component_parameters=None, args=None):
        self._init_runtime_parameters(component_parameters)
        data_sets = args["data"]

        data_statement_dict = list(data_sets.values())[0]
        need_eval = False
        for data_key in data_sets:
            if 'eval_data' in data_sets[data_key]:
                need_eval = True

        LOGGER.debug("data_sets: {}, data_statement_dict: {}".format(data_sets, data_statement_dict))
        if self.need_cv:
            LOGGER.info("Task is cross validation.")
            self.cross_validation(None)
            return

        elif not "model" in args:
            LOGGER.info("Task is fit")
            self.set_flowid('fit')
            self.fit()
            self.set_flowid('predict')
            self.predict()
            if need_eval:
                self.set_flowid('validate')
                self.predict()
        else:
            LOGGER.info("Task is predict")
            self._load_model(args)
            self.set_flowid('predict')
            self.predict()
