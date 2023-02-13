import math
from keras.callbacks import Callback
from keras import backend

# The SnapshotEnsemble class is partially attributed to Jason Brownlee as described in his blog reference below.
# However, this code has been significantly modified from his version for robustness. In particular,
# the following significant features have been added:
#
#    1) cosine-annealed learning rate avoids dropping to zero, which creates pathologic issues,
#    2) adds a warmup period where training does not begin cosine-annealing until after some 
#       number of epochs (default=5) have run.
#
# Currently, snapshots are saved off in the HDF5 keras format.
#
# Reference: https://machinelearningmastery.com/snapshot-ensemble-deep-learning-neural-network/
class SnapshotEnsemble(Callback):
    # constructor
    def __init__(self, filepath, iterationsPerCycle, lrate_max, warmupEpochs=5, verbose=False):
        self.filepath = filepath
        self.iterationsPerCycle = iterationsPerCycle
        self.lr_max = lrate_max
        self.warmupEpochs = warmupEpochs
        self.verbose = verbose
        self.lrates = list()
        self.iteration = 0
        self.snapshot = 0

    
    # calculate learning rate for iteration (within epoch)
    def cosine_annealing(self, iteration, lrate_max):
        if self.warmupEpochs > 0 :
            return lrate_max
        else :
            cos_inner = (math.pi * iteration) / max(self.iterationsPerCycle,iteration+1)
            alpha = 0.02 # Value dips to ~2% of the original value.
            scale = (math.cos(cos_inner) + 1.0)/2.0
            scale = (1.0-alpha)*scale + alpha
            return lrate_max * scale


    # calculate and set learning rate at the start of each iteration or batch
    def on_train_batch_begin(self, batch, logs=None):
        # calculate learning rate
        lr = self.cosine_annealing(self.iteration, self.lr_max)
        if self.warmupEpochs == 0 :
            self.iteration += 1
        # set learning rate
        backend.set_value(self.model.optimizer.lr, lr)
        # log value
        self.lrates.append(lr)


    # save models at the end of each cycle
    def on_epoch_end(self, epoch, logs={}):
        # First decrement warmupEpochs if necessary
        if self.warmupEpochs > 0 :
            self.warmupEpochs -= 1
            if self.verbose:
                print(f'>no snapshot saved at epoch {epoch}')
        # check if we can save model
        elif self.iteration >= self.iterationsPerCycle:
            # save model to file
            snapshot_epoch = self.snapshot
            self.snapshot += 1
            filename = self.filepath.format(**locals())
            if self.verbose: print(f'>Attempting to save snapshot {filename}, on epoch {epoch}')
            self.model.save(filename)
            print(f'>saved snapshot {filename}, on epoch {epoch}')
            self.iteration=0
        elif self.verbose:
            print(f'>no snapshot saved at epoch {epoch}')


