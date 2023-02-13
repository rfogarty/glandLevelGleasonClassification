# -*- coding: utf-8 -*-
"""
Created on Thur Dec 23 15:11:00 2021

@author: rfogarty
"""

import math
from keras.callbacks import CSVLogger,ModelCheckpoint,EarlyStopping

from data import *
from model import *
from snapshot import *
from archive import *
from gpuassign import *
from arguments import *
from dataParameters import *

def configCallbacks(args,paths) :
    #1 Configure Snapshot Ensemble (and Learning Rate Cosine Annealer)
    iterationsPerCycle = math.ceil((len(paths)/args.numPools)/args.bs) * args.cycle
    print(f'Iterations per Cycle: {iterationsPerCycle}')
    filepath = f"snapshot-weights-split{args.split}" + "-{snapshot_epoch:03d}.hdf5"
    checkpoint = SnapshotEnsemble(filepath,iterationsPerCycle,args.lr,verbose=True)
    #2 Configure Metrics Logger
    csv_logger = CSVLogger(f'log{args.split}.csv', append=True, separator=',')
    #3 Configure ModelCheckpoints
    bestfilepath = f"best-weights-split{args.split}" + "-{epoch:03d}.hdf5"
    cpMonitor,cpMode = checkpointMonitor()
    saveBest=ModelCheckpoint(filepath=bestfilepath,monitor=cpMonitor,mode=cpMode,save_best_only=True)
    #4 Lastly, configure EarlyStopping
    esMonitor,esMode=earlyStopMonitor()
    earlyStop=EarlyStopping(monitor=esMonitor,mode=esMode,min_delta=0.0005,patience=args.patience)
    
    callbacks_list = [checkpoint,csv_logger,saveBest,earlyStop]
    return callbacks_list

def trainModel(args) :
    mask_dir = None
    mask_ext = None
    if args.mask :
        mask_dir=maskDir()
        mask_ext=maskExt()

    dataIterator,labels,paths = getDataCutmixIteratorFromListing(dataDir(),trainingSet(args.split),
                                                                 blocklist=trainingBlocklists(args.split),
                                                                 shuffled=True,batch_size=args.bs,numPools=args.numPools,tilerank=args.rank,
                                                                 input_tensor_shape=args.tensor_shape,normalize=True,
                                                                 mask_dir=mask_dir,mask_ext=mask_ext)
    vdataIterator,vlabels,vpaths = getDataCutmixIteratorFromListing(dataDir(),validationSet(args.split),
                                                                 blocklist=validationBlocklists(args.split),
                                                                 shuffled=True,batch_size=args.bs,numPools=1,tilerank=args.rank,
                                                                 input_tensor_shape=args.tensor_shape,normalize=True,
                                                                 mask_dir=mask_dir,mask_ext=mask_ext)
    
    callbacks_list = configCallbacks(args,paths)
    
    model = buildTrainingNet(args.tensor_shape,numClasses(),gpuIds2Names(args.gpus),augment=args.augment)
    H = model.fit(dataIterator,batch_size=args.bs,
                  epochs=args.epochs,
                  validation_data=vdataIterator,
                  callbacks=callbacks_list, verbose=1)
    
    saveHistory(trainingHistory(args.split),H)
    
if __name__ == "__main__" :
    # Process arguments from commandline
    args = processCommandLine()

    # Limit which GPUs we can run on
    assignGPUs(args.gpus)

    # Train our model
    trainModel(args)
