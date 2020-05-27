import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pysnooper
import argparse
from functools import reduce

from ftl.encryption import paillier, encryption
from joblib import Parallel, delayed

tf.enable_eager_execution()

from tensorflow import contrib

tfe = contrib.eager


print("TensorFlow version: {}".format(tf.__version__))
# expected tensorflow 1.14
print("Eager execution: {}".format(tf.executing_eagerly()))

# load fashion mnist dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_labels = train_labels.astype(np.int32)
test_labels = test_labels.astype(np.int32)

train_images = train_images / 255.0
test_images = test_images / 255.0

def build_datasets(num_clients):

    # split_idx = int(len(x_train) / num_clients)
    avg_length = int(len(train_images) / num_clients)
    split_idx = [_ * avg_length for _ in range(1, num_clients)]

    # [train_images_0, train_images_1] = np.split(train_images, [split_idx])
    # [train_labels_0, train_labels_1] = np.split(train_labels, [split_idx])
    x_train_clients = np.split(train_images, split_idx)
    y_train_clients = np.split(train_labels, split_idx)


    # # party A
    # train_dataset_0 = tf.data.Dataset.from_tensor_slices((train_images_0, train_labels_0))
    # # party B
    # train_dataset_1 = tf.data.Dataset.from_tensor_slices((train_images_1, train_labels_1))
    train_dataset_clients = [tf.data.Dataset.from_tensor_slices(item) for item in zip(x_train_clients, y_train_clients)]

    BATCH_SIZE = 128
    SHUFFLE_BUFFER_SIZE = 100

    # train_dataset_0 = train_dataset_0.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    # train_dataset_1 = train_dataset_1.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    for i in range(len(train_dataset_clients)):
        train_dataset_clients[i] = train_dataset_clients[i].shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

    return train_dataset_clients


# build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.summary()
# predictions = model(features)
# print(predictions[:5])
cce = tf.keras.losses.SparseCategoricalCrossentropy()


def loss(model, x, y):
    y_ = model(x)
    return cce(y_true=y, y_pred=y_)


# l = loss(model, features, labels)
# print("Loss test: {}".format(l))


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
global_step = tf.Variable(0)


def clip_gradients(grads, min_v, max_v):
    results = [tf.clip_by_value(t, min_v, max_v) for t in grads]
    return results


def do_sum(x1, x2):
    results = []
    for i in range(len(x1)):
        results.append(x1[i] + x2[i])
    return results


def aggregate_gradients(gradient_list):
    results = reduce(do_sum, gradient_list)
    return results


def aggregate_losses(loss_list):
    return np.sum(loss_list)


def quantize(party, bit_width=16):
    result = []
    for component in party:
        x, _ = encryption.quantize_matrix(component, bit_width=bit_width)
        result.append(x)
    return np.array(result)


def quantize_per_layer(party, r_maxs, bit_width=16):
    result = []
    for component, r_max in zip(party, r_maxs):
        x, _ = encryption.quantize_matrix_stochastic(component, bit_width=bit_width, r_max=r_max)
        result.append(x)
    return np.array(result)


def unquantize(party, bit_width=16, r_max=0.5):
    result = []
    for component in party:
        result.append(encryption.unquantize_matrix(component, bit_width=bit_width, r_max=r_max).astype(np.float32))
    return np.array(result)


def unquantize_per_layer(party, r_maxs, bit_width=16):
    result = []
    for component, r_max in zip(party, r_maxs):
        result.append(encryption.unquantize_matrix(component, bit_width=bit_width, r_max=r_max).astype(np.float32))
    return np.array(result)


if __name__ == '__main__':
    seed = 123
    tf.random.set_random_seed(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    # parser.add_argument('--experiment', type=str, required=True,
    #                     choices=["plain", "batch", "only_quan", "aciq_quan"])
    parser.add_argument('--experiment', type=str, default="plain",
                        choices=["plain", "batch", "only_quan", "aciq_quan"])
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--q_width', type=int, default=16)
    parser.add_argument('--clip', type=float, default=0.5)
    args = parser.parse_args()

    options = vars(args)
    output_name = "fmnist_" + "_".join([ "{}_{}".format(key, options[key]) for key in options ])

    # keep results for plotting
    train_loss_results = []
    train_accuracy_results = []
    test_loss_results = []
    test_accuracy_results = []

    num_epochs  = args.num_epochs
    clip        = args.clip
    num_clients = args.num_clients
    q_width     = args.q_width

    # this key pair should be shared by party A and B
    publickey, privatekey = paillier.PaillierKeypair.generate_keypair(n_length=2048)

    loss_array = []
    accuracy_array = []


    for epoch in range(num_epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()

        train_dataset_clients = build_datasets(num_clients)

        for data_clients in zip(*train_dataset_clients):
            print("{} clients are in federated training".format(len(data_clients)))
            loss_batch_clients = []
            grads_batch_clients = []

            start_t = time.time()

            # calculate loss and grads locally
            for x, y in data_clients:
                loss_temp, grads_temp = grad(model, x, y)
                loss_batch_clients.append(loss_temp.numpy())
                grads_batch_clients.append([x.numpy() for x in grads_temp])

            # federated_lr_plain.py
            if args.experiment == "plain":
                # NOTE: The clip value here is "1" in federated_lr_plain.py
                # grads_0 = clip_gradients(grads_0, -1 * clip, clip)
                # grads_1 = clip_gradients(grads_1, -1 * clip, clip)

                # in plain version, no clipping before applying
                # grads_batch_clients = [clip_gradients(item, -1 * clip, clip) 
                #                         for item in grads_batch_clients]


                client_weight = 1.0 / num_clients
                start = time.time()
                grads      = aggregate_gradients(grads_batch_clients)
                end_enc = time.time()
                print("aggregation finished in %f" % (end_enc - start))
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])

            # federated_lr_batch.py
            elif args.experiment == "batch":
                # # party A
                # grads_0 = clip_gradients(grads_0, -1 * clip / num_clients, clip / num_clients)
                # # party B
                # grads_1 = clip_gradients(grads_1, -1 * clip / num_clients, clip / num_clients)
                grads_batch_clients = [clip_gradients(item, -1 * clip / num_clients, clip / num_clients) 
                                        for item in  grads_batch_clients]

                # # party A
                # enc_grads_0 = []
                # enc_grads_shape_0 = []
                # for component in grads_0:
                #     enc_g, enc_g_s = encryption.encrypt_matrix_batch(publickey, component.numpy(),
                #                                                      bit_width=q_width, r_max=clip)
                #     enc_grads_0.append(enc_g)
                #     enc_grads_shape_0.append(enc_g_s)
                # loss_value_0 = encryption.encrypt(publickey, loss_value_0)
                # # party B
                # enc_grads_1 = []
                # enc_grads_shape_1 = []
                # for component in grads_1:
                #     enc_g, enc_g_s = encryption.encrypt_matrix_batch(publickey, component.numpy(),
                #                                                      bit_width=q_width, r_max=clip)
                #     enc_grads_1.append(enc_g)
                #     enc_grads_shape_1.append(enc_g_s)
                # loss_value_1 = encryption.encrypt(publickey, loss_value_1)
                enc_grads_batch_clients = []
                enc_grads_shape_batch_clients = []
                for grad_client in grads_batch_clients:
                    enc_grads_client = []
                    enc_grads_shape_client = []
                    for component in grad_client:
                        enc_g, enc_g_s = encryption.encrypt_matrix_batch(publickey, component.numpy(),
                                                                         bit_width=q_width, r_max=clip)
                        enc_grads_client.append(enc_g)
                        enc_grads_shape_client.append(enc_g_s)

                    enc_grads_batch_clients.append(enc_grads_client)
                    enc_grads_shape_batch_clients.append(enc_grads_shape_client)

                loss_batch_clients = [encryption.encrypt(publickey, item) for item in loss_batch_clients] 

                # arbiter aggregate gradients
                # enc_grads = aggregate_gradients([enc_grads_0, enc_grads_1])
                # loss_value = aggregate_losses([loss_value_0, loss_value_1])
                enc_grads = aggregate_gradients(enc_grads_batch_clients)
                client_weight = 1.0 / num_clients
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])

                # on party A and B individually
                loss_value = encryption.decrypt(privatekey, loss_value)
                grads = []
                for i in range(len(enc_grads)):
                    # plain_g = encryption.decrypt_matrix_batch(privatekey, enc_grads_0[i], enc_grads_shape_0[i])
                    plain_g = encryption.decrypt_matrix_batch(privatekey, enc_grads[i], enc_grads_shape_batch_clients[0][i])
                    grads.append(plain_g)

            # federated_lr_only_quan.py
            elif args.experiment == "only_quan":
                for idx in range(len(grads_batch_clients)):
                    grads_batch_clients[idx] = [x.numpy() for x in grads_batch_clients[idx]]

                # clipping_thresholds = encryption.calculate_clip_threshold(grads_0)
                theta = 2.5
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

                # r_maxs = [x * 2 for x in clipping_thresholds]
                r_maxs = [x * num_clients for x in clipping_thresholds]

             	# grads_0 = encryption.clip_with_threshold(grads_0, clipping_thresholds)
            	# grads_1 = encryption.clip_with_threshold(grads_1, clipping_thresholds)
            	# grads_0 = quantize_per_layer(grads_0, r_maxs, bit_width=q_width)
            	# grads_1 = quantize_per_layer(grads_1, r_maxs, bit_width=q_width)
                grads_batch_clients = [encryption.clip_with_threshold(item, clipping_thresholds)
                                       for item in grads_batch_clients]

                grads_batch_clients = [quantize_per_layer(item, r_maxs, bit_width=q_width) 
                                       for item in grads_batch_clients]

                # grads = aggregate_gradients([grads_0, grads_1], weight=0.5)
                # loss_value = aggregate_losses([0.5 * loss_value_0, 0.5 * loss_value_1])

               # grads = unquantize_per_layer(grads, r_maxs, bit_width=q_width)
                client_weight = 1.0 / num_clients
                grads      = aggregate_gradients(grads_batch_clients)
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])

                grads = unquantize_per_layer(grads, r_maxs, bit_width=q_width)

            elif args.experiment == "aciq_quan":
                sizes = [tf.size(item).numpy() * num_clients for item in grads_batch_clients[0]]
                max_values = []
                min_values = []
                for layer_idx in range(len(grads_batch_clients[0])):
                    max_values.append([np.max([item[layer_idx] for item in grads_batch_clients])])
                    min_values.append([np.min([item[layer_idx] for item in grads_batch_clients])])
                grads_max_min = np.concatenate([np.array(max_values),np.array(min_values)], axis=1)
                clipping_thresholds = encryption.calculate_clip_threshold_aciq_g(grads_max_min, sizes, bit_width=q_width)

                r_maxs = [x * num_clients for x in clipping_thresholds]
                grads_batch_clients = [encryption.clip_with_threshold(item, clipping_thresholds)
                                       for item in grads_batch_clients]
                grads_batch_clients = [quantize_per_layer(item, r_maxs, bit_width=q_width)
                                       for item in grads_batch_clients]

                grads = aggregate_gradients(grads_batch_clients)
                client_weight = 1.0 / num_clients
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])

                grads = unquantize_per_layer(grads, r_maxs, bit_width=q_width)

            ######
            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                      global_step)

            # Track progress
            epoch_loss_avg(loss_value)  # add current batch loss
            # compare predicted label to actual label
            # epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)
            epoch_accuracy(tf.argmax(model(test_images), axis=1, output_type=tf.int32),
                           test_labels)

            loss_array.append(loss_value)
            accuracy_value = epoch_accuracy.result().numpy()
            accuracy_array.append(accuracy_value)

            elapsed_time = time.time() - start_t
            print( "loss: {} \taccuracy: {} \telapsed time: {}".format(loss_value, accuracy_value, elapsed_time) )
            
        # end epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        test_loss_v = loss(model, test_images, test_labels)
        test_accuracy_v = accuracy_score(test_labels, tf.argmax(model(test_images), axis=1, output_type=tf.int32))
        test_loss_results.append(test_loss_v)
        test_accuracy_results.append(test_accuracy_v)

        if epoch % 1 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))
        model.save_weights("model_{}_e{:03d}.h5".format(output_name, epoch))
        print("Saved model to disk")

    np.savetxt('train_loss_{}.txt'.format(output_name), train_loss_results)
    np.savetxt('train_accuracy_{}.txt'.format(output_name), train_accuracy_results)
    np.savetxt('test_loss_{}.txt'.format(output_name), test_loss_results)
    np.savetxt('test_accuracy_{}.txt'.format(output_name), test_accuracy_results)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model_{}.json".format(output_name), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_{}.h5".format(output_name))
    print("Saved model to disk")



fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()
