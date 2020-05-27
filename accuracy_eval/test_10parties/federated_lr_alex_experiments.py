import time
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import accuracy_score
import pysnooper

from functools import reduce

from ftl import augmentation
from ftl.encryption import paillier, encryption

tf.enable_eager_execution()

from tensorflow import contrib

from joblib import Parallel, delayed

import multiprocessing

N_JOBS = multiprocessing.cpu_count()

tfe = contrib.eager

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# epochs = 200
num_classes = 10

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# x_train = x_train[:1000]
# y_train = y_train[:1000]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = y_train.squeeze().astype(np.int32)
y_test = y_test.squeeze().astype(np.int32)


def build_datasets(num_clients):
    # split_idx = int(len(x_train) / num_clients)
    avg_length = int(len(x_train) / num_clients)
    split_idx = [_ * avg_length for _ in range(1, num_clients)]

    # [x_train_0, x_train_1] = np.split(x_train, [split_idx])
    # [y_train_0, y_train_1] = np.split(y_train, [split_idx])
    x_train_clients = np.split(x_train, split_idx)
    y_train_clients = np.split(y_train, split_idx)

    print("{} clients building datasets.".format(len(x_train_clients)))
    for idx, x_train_client in enumerate(x_train_clients):
        print("{} client has {} data items.".format(idx, len(x_train_client)))

    # train_dataset_0 = tf.data.Dataset.from_tensor_slices((x_train_0, y_train_0))
    # train_dataset_1 = tf.data.Dataset.from_tensor_slices((x_train_1, y_train_1))
    train_dataset_clients = [tf.data.Dataset.from_tensor_slices(item) for item in zip(x_train_clients, y_train_clients)]

    # train_dataset_0 = train_dataset_0.map(
    #         augmentation.augment_img,
    #         num_parallel_calls=N_JOBS)

    # train_dataset_0 = train_dataset_0.map(lambda x, y: (tf.clip_by_value(x, 0, 1), y))

    # train_dataset_1 = train_dataset_1.map(
    #     augmentation.augment_img,
    #     num_parallel_calls=N_JOBS)
    # train_dataset_1 = train_dataset_1.map(lambda x, y: (tf.clip_by_value(x, 0, 1), y))

    BATCH_SIZE = 128
    SHUFFLE_BUFFER_SIZE = 1000

    # train_dataset_0 = train_dataset_0.shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE)
    # train_dataset_1 = train_dataset_1.shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE)

    for i in range(len(train_dataset_clients)):
        train_dataset_clients[i] = train_dataset_clients[i].map(
            augmentation.augment_img,
            num_parallel_calls=N_JOBS)
        train_dataset_clients[i] = train_dataset_clients[i].map(lambda x, y: (tf.clip_by_value(x, 0, 1), y))
        train_dataset_clients[i] = train_dataset_clients[i].shuffle(SHUFFLE_BUFFER_SIZE,
                                                                    reshuffle_each_iteration=True).batch(BATCH_SIZE)

    # return train_dataset_0, train_dataset_1
    return train_dataset_clients


model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), padding='same',
                              input_shape=x_train.shape[1:]))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(32, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(64, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes))
model.add(keras.layers.Activation('softmax'))
model.summary()

cce = tf.keras.losses.SparseCategoricalCrossentropy()


def loss(model, x, y):
    y_ = model(x)
    return cce(y_true=y, y_pred=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


optimizer = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
global_step = tf.Variable(0)


def clip_gradients(grads, min_v, max_v):
    results = [tf.clip_by_value(t, min_v, max_v).numpy() for t in grads]
    return results


def do_sum(x1, x2):
    results = []
    for i in range(len(x1)):
        results.append(x1[i] + x2[i])
    return results


def aggregate_gradients(gradient_list, weight=0.5):
    # def multiply_by_weight(party, w):
    #     for i in range(len(party)):
    #         party[i] = w * party[i]
    #     return party

    # gradient_list = Parallel(n_jobs=2)(delayed(multiply_by_weight)(party, weight) for party in gradient_list)
    results = reduce(do_sum, gradient_list)
    return results


def aggregate_losses(loss_list):
    return np.sum(loss_list)


def batch_enc_per_layer(publickey, party, r_maxs, bit_width=16, batch_size=100):
    result = []
    og_shapes = []

    for layer, r_max in zip(party, r_maxs):
        enc, shape_ = encryption.encrypt_matrix_batch(publickey, layer, batch_size=batch_size, bit_width=bit_width,
                                                      r_max=r_max)
        result.append(enc)
        og_shapes.append(shape_)
    return result, og_shapes


def batch_dec_per_layer(privatekey, party, og_shapes, r_maxs, bit_width=16, batch_size=100):
    result = []
    for layer, r_max, og_shape in zip(party, r_maxs, og_shapes):
        result.append(
            encryption.decrypt_matrix_batch(privatekey, layer, og_shape, batch_size=batch_size, bit_width=bit_width,
                                            r_max=r_max).astype(np.float32))
    return np.array(result)


def quantize(party, bit_width=16, r_max=0.5):
    result = []
    for component in party:
        x, _ = encryption.quantize_matrix(component, bit_width=bit_width, r_max=r_max)
        result.append(x)
    return np.array(result)


def quantize_per_layer(party, r_maxs, bit_width=16):
    # result = []
    # for component, r_max in zip(party, r_maxs):
    #     x, _ = encryption.quantize_matrix_stochastic(component, bit_width=bit_width, r_max=r_max)
    #     result.append(x)
    result = Parallel(n_jobs=N_JOBS)(
        delayed(encryption.quantize_matrix_stochastic)(component, bit_width=bit_width, r_max=r_max) for component, r_max
        in zip(party, r_maxs))
    result = np.array(result)[:, 0]
    return result


def unquantize(party, bit_width=16, r_max=0.5):
    result = []
    for component in party:
        result.append(encryption.unquantize_matrix(component, bit_width=bit_width, r_max=r_max).astype(np.float32))
    return np.array(result)


def unquantize_per_layer(party, r_maxs, bit_width=16):
    # result = []
    # for component, r_max in zip(party, r_maxs):
    #     result.append(encryption.unquantize_matrix(component, bit_width=bit_width, r_max=r_max).astype(np.float32))
    result = Parallel(n_jobs=N_JOBS)(
        delayed(encryption.unquantize_matrix)(component, bit_width=bit_width, r_max=r_max) for component, r_max
        in zip(party, r_maxs)
    )
    return np.array(result)


if __name__ == '__main__':
    seed = 123
    tf.random.set_random_seed(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='plain_alex',
                        choices=["plain_alex", "plain_alex_en", "en_alex_batch", "aciq_quan"])
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--q_width', type=int, default=16)
    args = parser.parse_args()
    print(args)
    # keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    test_loss_results = []
    test_accuracy_results = []

    num_epochs = args.num_epochs

    publickey, privatekey = paillier.PaillierKeypair.generate_keypair(n_length=2048)

    for epoch in range(num_epochs):
        epoch_train_loss_avg = tfe.metrics.Mean()
        epoch_train_accuracy = tfe.metrics.Accuracy()

        # train_dataset_0, train_dataset_1 = build_datasets()
        train_dataset_clients = build_datasets(args.num_clients)

        # for (x_0, y_0), (x_1, y_1) in zip(train_dataset_0, train_dataset_1):
        for data_clients in zip(*train_dataset_clients):
            print("{} clients are in federated training".format(len(data_clients)))
            loss_batch_clients = []
            grads_batch_clients = []

            start_t = time.time()
            # Optimize the model
            # loss_value_0, grads_0 = grad(model, x_0, y_0)
            # loss_value_1, grads_1 = grad(model, x_1, y_1)

            # loss_value_0 = loss_value_0.numpy()
            # loss_value_1 = loss_value_1.numpy()

            # grads = [x.numpy() for x in grads]

            # grads_0 = [x.numpy() for x in grads_0]
            # grads_1 = [x.numpy() for x in grads_1]

            # calculate loss and grads locally
            for x, y in data_clients:
                loss_temp, grads_temp = grad(model, x, y)
                loss_batch_clients.append(loss_temp.numpy())
                grads_batch_clients.append([x.numpy() for x in grads_temp])

            if args.experiment == "plain_alex":
                start = time.time()
                grads = aggregate_gradients(grads_batch_clients)
                end_enc = time.time()
                print("aggregation finished in %f" % (end_enc - start))
                client_weight = 1.0 / args.num_clients
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])

            elif args.experiment == "plain_alex_en":
                # grads_0 = [encryption.encrypt_matrix(publickey, x) for x in grads_0]
                # grads_1 = [encryption.encrypt_matrix(publickey, x) for x in grads_1]
                # loss_value_0 = encryption.encrypt(publickey, loss_value_0)
                # loss_value_1 = encryption.encrypt(publickey, loss_value_1)

                loss_batch_clients = [encryption.encrypt(publickey, item)
                                      for item in loss_batch_clients]
                grads_batch_clients = [[encryption.encrypt_matrix(publickey, x) for x in item]
                                       for item in grads_batch_clients]
                # grads = aggregate_gradients([grads_0, grads_1])
                # loss_value = aggregate_losses([0.5 * loss_value_0, 0.5 * loss_value_1])
                grads = aggregate_gradients(grads_batch_clients)
                client_weight = 1.0 / args.num_clients
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])

                loss_value = encryption.decrypt(privatekey, loss_value)
                grads = [encryption.decrypt_matrix(privatekey, x).astype(np.float32) for x in grads]

            # federated_lr_en_alex_batch.py part
            elif args.experiment == "en_alex_batch":
                theta = 2.5
                # clipping_thresholds = encryption.calculate_clip_threshold(grads_0) return [theta * np.std(x) for x in grads]
                # theta = 2.5
                # calculate global std by combination, clients send E(X^2), E(X), and layerwise sizes to the server
                # std = E(X^2) - (E(X))^2

                grads_batch_clients_mean = []
                grads_batch_clients_mean_square = []
                for client_idx in range(len(grads_batch_clients)):
                    temp_mean = [np.mean(grads_batch_clients[client_idx][layer_idx])
                                 for layer_idx in range(len(grads_batch_clients[client_idx]))]
                    temp_mean_square = [np.mean(grads_batch_clients[client_idx][layer_idx] ** 2)
                                        for layer_idx in range(len(grads_batch_clients[client_idx]))]
                    grads_batch_clients_mean.append(temp_mean)
                    grads_batch_clients_mean_square.append(temp_mean_square)
                grads_batch_clients_mean = np.array(grads_batch_clients_mean)
                grads_batch_clients_mean_square = np.array(grads_batch_clients_mean_square)

                layers_size = np.array([_.size for _ in grads_batch_clients[0]])
                clipping_thresholds = theta * (
                            np.sum(grads_batch_clients_mean_square * layers_size, 0) / (layers_size * num_clients)
                            - (np.sum(grads_batch_clients_mean * layers_size, 0) / (layers_size * num_clients)) ** 2) ** 0.5

                print("clipping_thresholds", clipping_thresholds)

                r_maxs = [x * args.num_clients for x in clipping_thresholds]

                # grads_0 = encryption.clip_with_threshold(grads_0, clipping_thresholds)
                # grads_1 = encryption.clip_with_threshold(grads_1, clipping_thresholds)
                grads_batch_clients = [encryption.clip_with_threshold(item, clipping_thresholds)
                                       for item in grads_batch_clients]

                # enc_grads_0, og_shape_0 = batch_enc_per_layer(publickey=publickey, party=grads_0, r_maxs=r_maxs, bit_width=q_width,
                #                                               batch_size=batch_size)
                # enc_grads_1, og_shape_1 = batch_enc_per_layer(publickey=publickey, party=grads_1, r_maxs=r_maxs, bit_width=q_width,
                #                                               batch_size=batch_size)
                enc_grads_batch_clients = []
                og_shape_batch_clients = []
                for item in grads_batch_clients:
                    enc_grads_temp, og_shape_temp = batch_enc_per_layer(publickey=publickey, party=item,
                                                                        r_maxs=r_maxs,
                                                                        bit_width=args.q_width,
                                                                        batch_size=args.batch_size)
                    enc_grads_batch_clients.append(enc_grads_temp)
                    og_shape_batch_clients.append(og_shape_temp)

                # loss_value_0 = publickey.encrypt(loss_value_0)
                # loss_value_1 = publickey.encrypt(loss_value_1)
                loss_batch_clients = [publickey.encrypt(item) for item in loss_batch_clients]

                # grads = aggregate_gradients([enc_grads_0, enc_grads_1])
                # loss_value = aggregate_losses([0.5 * loss_value_0, 0.5 * loss_value_1])
                grads = aggregate_gradients(enc_grads_batch_clients)
                client_weight = 1.0 / args.num_clients
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])

                # loss_value = encryption.decrypt(privatekey, loss_value)
                # grads = batch_dec_per_layer(privatekey=privatekey, party=grads, og_shapes=og_shape_0, r_maxs=r_maxs, bit_width=q_width, batch_size=batch_size)
                loss_value = encryption.decrypt(privatekey, loss_value)
                grads = batch_dec_per_layer(privatekey=privatekey, party=grads, og_shapes=og_shape_batch_clients[0],
                                            r_maxs=r_maxs, bit_width=args.q_width, batch_size=args.batch_size)

            # federated_lr_quan_alex.py
            elif args.experiment == "aciq_quan":

                sizes = [item.size * args.num_clients for item in grads_batch_clients[0]]
                max_values = []
                min_values = []
                for layer_idx in range(len(grads_batch_clients[0])):
                    max_values.append([np.max([item[layer_idx] for item in grads_batch_clients])])
                    min_values.append([np.min([item[layer_idx] for item in grads_batch_clients])])
                grads_max_min = np.concatenate([np.array(max_values),np.array(min_values)],axis=1)
                clipping_thresholds = encryption.calculate_clip_threshold_aciq_g(grads_max_min, sizes, bit_width=args.q_width)


                r_maxs = [x * args.num_clients for x in clipping_thresholds]

                # grads_0 = encryption.clip_with_threshold(grads_0, clipping_thresholds)
                # grads_1 = encryption.clip_with_threshold(grads_1, clipping_thresholds)
                grads_batch_clients = [encryption.clip_with_threshold(item, clipping_thresholds)
                                       for item in grads_batch_clients]

                # grads_0 = quantize_per_layer(grads_0, r_maxs, bit_width=q_width)
                # grads_1 = quantize_per_layer(grads_1, r_maxs, bit_width=q_width)
                grads_batch_clients = [quantize_per_layer(item, r_maxs, bit_width=args.q_width)
                                       for item in grads_batch_clients]

                # grads = aggregate_gradients([grads_0, grads_1])
                # loss_value = aggregate_losses([0.5 * loss_value_0, 0.5 * loss_value_1])
                grads = aggregate_gradients(grads_batch_clients)
                client_weight = 1.0 / args.num_clients
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])

                # grads = unquantize_per_layer(grads, r_maxs, bit_width=q_width)
                grads = unquantize_per_layer(grads, r_maxs, bit_width=args.q_width)

            #######
            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                      global_step)

            elapsed_time = time.time() - start_t
            print('loss: {} \telapsed time: {}'.format(loss_value, elapsed_time))

            # Track progress
            epoch_train_loss_avg(loss_value)  # add current batch loss
            # compare predicted label to actual label
            # epoch_train_accuracy(tf.argmax(model(tf.concat([x_0, x_1], axis=0)), axis=1, output_type=tf.int32),
            #                      tf.concat([y_0, y_1], axis=0))
            epoch_train_accuracy(
                tf.argmax(model(tf.concat([data_item[0] for data_item in data_clients], axis=0)), axis=1,
                          output_type=tf.int32),
                tf.concat([data_item[1] for data_item in data_clients], axis=0))

        # end epoch
        train_loss_v = epoch_train_loss_avg.result()
        train_accuracy_v = epoch_train_accuracy.result()
        test_loss_v = loss(model, x_test, y_test)
        test_accuracy_v = accuracy_score(y_test, tf.argmax(model(x_test), axis=1, output_type=tf.int32))

        train_loss_results.append(train_loss_v)
        train_accuracy_results.append(train_accuracy_v)
        test_loss_results.append(test_loss_v)
        test_accuracy_results.append(test_accuracy_v)

        print(
            "Epoch {:03d}: train_loss: {:.3f}, train_accuracy: {:.3%}, test_loss: {:.3f}, test_accuracy: {:.3%}".format(
                epoch,
                train_loss_v,
                train_accuracy_v,
                test_loss_v,
                test_accuracy_v))

        # serialize weights to HDF5
        model.save_weights("model_alex_b{:02d}_{}_e{:03d}.h5".format(args.q_width, args.experiment, epoch))
        print("Saved model to disk")

    np.savetxt('alex_v_train_loss_b{:02d}_final_{}.txt'.format(args.q_width, args.experiment), train_loss_results)
    np.savetxt('alex_v_train_accuracy_b{:02d}_final_{}.txt'.format(args.q_width, args.experiment),
               train_accuracy_results)
    np.savetxt('alex_v_test_loss_b{:02d}_final_{}.txt'.format(args.q_width, args.experiment), test_loss_results)
    np.savetxt('alex_v_test_accuracy_b{:02d}_final_{}.txt'.format(args.q_width, args.experiment), test_accuracy_results)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model_alex_b{:02d}_{}.json".format(args.q_width, args.experiment), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_alex_b{:02d}_{}.h5".format(args.q_width, args.experiment))
    print("Saved model to disk")
