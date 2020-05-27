import time
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pysnooper
import copy 

from functools import reduce, partial

from ftl.encryption import paillier, encryption

tf.enable_eager_execution()

from tensorflow import contrib

from joblib import Parallel, delayed

import multiprocessing

N_JOBS = multiprocessing.cpu_count()

tfe = contrib.eager

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                       'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print('Length of text: {} characters'.format(len(text)))

# print(text[:250])

# The unique characters in the file
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))

# Creating a mapping from unique characters to indices
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

print('{')
for char, _ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')

print('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text) // seq_length

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])

sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset_0 = sequences.map(split_input_target)

for input_example, target_example in dataset_0.take(1):
    print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print('Target data:', repr(''.join(idx2char[target_example.numpy()])))

for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

# Batch size
BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch // BATCH_SIZE

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset_0 = dataset_0.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

if tf.test.is_gpu_available():
    rnn = tf.keras.layers.CuDNNGRU
else:
    rnn = partial(
        tf.keras.layers.GRU, recurrent_activation='sigmoid')


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        rnn(rnn_units,
            return_sequences=True,
            recurrent_initializer='glorot_uniform',
            stateful=True),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)
model.summary()

for input_example_batch, target_example_batch in dataset_0.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))


def build_datasets(num_clients):
    dataset_raw = sequences.map(split_input_target)
    train_dataset_clients = [dataset_raw.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
                                for _ in range(num_clients)]

    return train_dataset_clients

def loss(model, x, y):
    y_ = model(x)
    return tf.keras.losses.sparse_categorical_crossentropy(y, y_, from_logits=True)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


optimizer = tf.train.AdamOptimizer()
global_step = tf.Variable(0)


def clip_gradients(grads, min_v, max_v):
    results = [tf.clip_by_value(t, min_v, max_v).numpy() for t in grads]
    return results


def do_sum(x1, x2):
    results = []
    for i in range(len(x1)):
        if isinstance(x1[i], tf.IndexedSlices) and isinstance(x2[i], tf.IndexedSlices):
            results.append(tf.IndexedSlices(values=tf.concat([x1[i].values, x2[i].values], axis=0),
                                            indices=tf.concat([x1[i].indices, x2[i].indices], axis=0),
                                            dense_shape=x1[i].dense_shape))
        else:
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
    return np.mean(loss_list)


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


def unquantize_per_layer(party, r_maxs, bit_width=16):
    # result = []
    # for component, r_max in zip(party, r_maxs):
    #     result.append(encryption.unquantize_matrix(component, bit_width=bit_width, r_max=r_max).astype(np.float32))
    result = Parallel(n_jobs=N_JOBS)(
        delayed(encryption.unquantize_matrix)(component, bit_width=bit_width, r_max=r_max) for component, r_max
        in zip(party, r_maxs)
    )
    return np.array(result)


def sparse_to_dense(gradients):
    result = []
    for layer in gradients:
        if isinstance(layer, tf.IndexedSlices):
            result.append(tf.convert_to_tensor(layer).numpy())
        else:
            result.append(layer.numpy())
    return result

if __name__ == '__main__':
    seed = 123
    tf.random.set_random_seed(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='plain',
                        choices=["plain", "quan", "aciq_quan"])
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--q_width', type=int, default=16)
    parser.add_argument('--clip', type=float, default=0.5)
    args = parser.parse_args()

    options = vars(args)
    output_name = "lstm_" + "_".join([ "{}_{}".format(key, options[key]) for key in options ])

    num_epochs  = args.num_epochs
    clip        = args.clip
    num_clients = args.num_clients
    q_width     = args.q_width

    # keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    # num_epochs = 100

    publickey, privatekey = paillier.PaillierKeypair.generate_keypair(n_length=2048)


    # with pysnooper.snoop('no_batch_log_sto_c05.log'):
    for epoch in range(num_epochs):
        epoch_loss_avg = tfe.metrics.Mean()

        # train_dataset_1_iter = iter(dataset_0)
        train_dataset_clients = build_datasets(num_clients)

        # for x_0, y_0 in dataset_0:
        #     x_1, y_1 = next(train_dataset_1_iter)
        for data_clients in zip(*train_dataset_clients):
            print("{} clients are in federated training".format(len(data_clients)))
            loss_batch_clients = []
            grads_batch_clients = []

            start_t = time.time()

            # # Optimize the model
            # loss_value_0, grads_0 = grad(model, x_0, y_0)
            # loss_value_1, grads_1 = grad(model, x_1, y_1)

            # calculate loss and grads locally
            for x, y in data_clients:
                loss_temp, grads_temp = grad(model, x, y)
                loss_batch_clients.append(loss_temp.numpy())
                grads_batch_clients.append(grads_temp)

            # federated_lr_plain_lstm.py
            if args.experiment == "plain":

                # loss_value_0 = loss_value_0.numpy()
                # loss_value_1 = loss_value_1.numpy()

                print

                # grads = aggregate_gradients([grads_0, grads_1])
                # loss_value = aggregate_losses([0.5 * loss_value_0, 0.5 * loss_value_1])
                start = time.time()
                grads = aggregate_gradients(grads_batch_clients)
                end_enc = time.time()
                print("aggregation finished in %f" % (end_enc - start))
                client_weight = 1.0 / num_clients
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])

            # federated_lr_quan_lstm.py
            elif args.experiment == "quan":
                # clipping_thresholds = encryption.calculate_clip_threshold_sparse(grads_0)
                theta = 2.5
                grads_batch_clients_mean = []
                grads_batch_clients_mean_square = []
                for client_idx in range(len(grads_batch_clients)):
                    temp_mean = []
                    temp_mean_square = []

                    for layer_idx in range(len(grads_batch_clients[client_idx])):
                        if isinstance(grads_batch_clients[client_idx][layer_idx], tf.IndexedSlices):
                            temp_mean.append( np.mean(grads_batch_clients[client_idx][layer_idx].values.numpy()) )
                            temp_mean_square.append( np.mean(grads_batch_clients[client_idx][layer_idx].values.numpy()**2) )
                        else:
                            temp_mean.append( np.mean(grads_batch_clients[client_idx][layer_idx].numpy()) )
                            temp_mean_square.append( np.mean(grads_batch_clients[client_idx][layer_idx].numpy()**2) )

                    grads_batch_clients_mean.append(temp_mean)
                    grads_batch_clients_mean_square.append(temp_mean_square)
                grads_batch_clients_mean = np.array(grads_batch_clients_mean)
                grads_batch_clients_mean_square = np.array(grads_batch_clients_mean_square)

                layers_size = []
                for layer in grads_batch_clients[0]:
                    if isinstance(layer, tf.IndexedSlices):
                        layers_size.append(layer.values.numpy().size)
                    else:
                        layers_size.append(layer.numpy().size)
                layers_size = np.array(layers_size)

                clipping_thresholds = theta * (
                            np.sum(grads_batch_clients_mean_square * layers_size, 0) / (layers_size * num_clients)
                            - (np.sum(grads_batch_clients_mean * layers_size, 0) / (layers_size * num_clients)) ** 2) ** 0.5

                print("clipping_thresholds", clipping_thresholds)


                # r_maxs = [x * 2 for x in clipping_thresholds]
                r_maxs = [x * num_clients for x in clipping_thresholds]

                # grads_0 = encryption.clip_with_threshold(sparse_to_dense(grads_0), clipping_thresholds)
                # grads_1 = encryption.clip_with_threshold(sparse_to_dense(grads_1), clipping_thresholds)
                grads_batch_clients = [encryption.clip_with_threshold(sparse_to_dense(item), clipping_thresholds)
                                        for item in grads_batch_clients]

                # grads_0 = quantize_per_layer(grads_0, r_maxs, bit_width=q_width)
                # grads_1 = quantize_per_layer(grads_1, r_maxs, bit_width=q_width)
                grads_batch_clients = [quantize_per_layer(item, r_maxs, bit_width=q_width) 
                                        for item in grads_batch_clients]

                # grads = aggregate_gradients([grads_0, grads_1])
                # loss_value = aggregate_losses([0.5 * loss_value_0, 0.5 * loss_value_1])
                grads = aggregate_gradients(grads_batch_clients)
                client_weight = 1.0 / num_clients
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])

                grads = unquantize_per_layer(grads, r_maxs, bit_width=q_width)

            elif args.experiment == "aciq_quan":
                grads_batch_clients_values = copy.deepcopy(grads_batch_clients)
                for client_idx in range(len(grads_batch_clients)):
                    for layer_idx in range(len(grads_batch_clients[client_idx])):
                        if isinstance(grads_batch_clients[client_idx][layer_idx], tf.IndexedSlices):
                            grads_batch_clients_values[client_idx][layer_idx] = grads_batch_clients[client_idx][layer_idx].values.numpy()
                        else:
                            grads_batch_clients_values[client_idx][layer_idx] = grads_batch_clients[client_idx][layer_idx].numpy()
                sizes = [item.size * num_clients for item in grads_batch_clients_values[0]]
                max_values = []
                min_values = []
                for layer_idx in range(len(grads_batch_clients_values[0])):
                    max_values.append([np.max([item[layer_idx] for item in grads_batch_clients_values])])
                    min_values.append([np.min([item[layer_idx] for item in grads_batch_clients_values])])
                grads_max_min = np.concatenate([np.array(max_values),np.array(min_values)], axis=1)
                clipping_thresholds = encryption.calculate_clip_threshold_aciq_g(grads_max_min, sizes, bit_width=q_width)
            
                r_maxs = [x * num_clients for x in clipping_thresholds]

                grads_batch_clients = [encryption.clip_with_threshold(sparse_to_dense(item), clipping_thresholds)
                                        for item in grads_batch_clients]
                grads_batch_clients = [quantize_per_layer(item, r_maxs, bit_width=q_width) 
                                        for item in grads_batch_clients]

                grads = aggregate_gradients(grads_batch_clients)
                client_weight = 1.0 / num_clients
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])

                grads = unquantize_per_layer(grads, r_maxs, bit_width=q_width)

            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                      global_step)

            # Track progress
            epoch_loss_avg(loss_value)  # add current batch loss
            # compare predicted label to actual label
            # epoch_accuracy(tf.argmax(model(tf.concat([x_0, x_1], axis=0)), axis=1, output_type=tf.int32),
            #                tf.concat([y_0, y_1], axis=0))

            elapsed_time = time.time() - start_t
            print( "loss: {} \telapsed time: {}".format(loss_value, elapsed_time) )

        # end epoch
        train_loss_results.append(epoch_loss_avg.result())

        print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))

        # checkpoint
        # model.save_weights( 'plain/easy_checkpoint_{:03d}'.format(epoch) )
        model.save_weights( '{}/easy_checkpoint_{:03d}'.format(args.experiment, epoch) )

    # np.savetxt('alex_v_loss_plain_lstm.txt', train_loss_results)
    np.savetxt('lstm_v_loss_{}.txt'.format(output_name), train_loss_results)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model_{}.json".format(output_name), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_{}.h5".format(output_name))
    print("Saved model to disk")

    # diskfig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    # fig.suptitle('Training Metrics')
    #
    # axes[0].set_ylabel("Loss", fontsize=14)
    # axes[0].plot(loss_array)
    #
    # axes[1].set_ylabel("Accuracy", fontsize=14)
    # axes[1].set_xlabel("Batch", fontsize=14)
    # axes[1].plot(accuracy_array)
    # plt.show()
