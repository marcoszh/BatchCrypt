import numpy as np

# We create a simple class, called EarlyStoppingCheckPoint, that combines simplified version of Early Stoping and ModelCheckPoint classes from Keras.
# References:
# https://keras.io/callbacks/
# https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L458
# https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L358


class EarlyStoppingCheckPoint(object):

    def __init__(self, monitor, patience, file_path=None):

        self.model = None
        self.monitor = monitor
        self.patience = patience
        self.file_path = file_path
        self.wait = 0
        self.stopped_epoch = 0

    def set_model(self, model):
        self.model = model

    def on_train_begin(self):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf

    def on_iteration_end(self, epoch, batch, logs=None):

        current = logs.get(self.monitor)
        if current is None:
            print('monitor does not available in logs')
            return

        if current > self.best:
            self.best = current
            print("find best acc: ", self.best, "at epoch:", epoch, "batch:", batch)
            if self.file_path is not None:
                self.model.save_model(self.file_path)
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
