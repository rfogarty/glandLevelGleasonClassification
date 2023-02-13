import argparse
from dataParameters import *

def addCommonArguments(parser,BS) :
    img_rows, img_cols, img_channels = tensorSize()
    tileRank=3
    numPools=1
    # Image control
    prescale = 1.0

    parser.add_argument('-g','--gpus',dest='gpus',type=int,nargs='+',required=False,default=(0),
                        help='split for this training',metavar='GPUS')
    parser.add_argument('-b','--bs',dest='bs',type=int,required=False,default=BS,
                        help='Batch size when training',metavar='BATCH')
    parser.add_argument('-P','--numPools',dest='numPools',type=int,required=False,default=numPools,
                        help='Number of Pools to split the data each epoch',metavar='POOLS')

    parser.add_argument('-t','--tensor-shape',dest='tensor_shape',type=int,nargs=3,required=False,
                        help='tensor shape for input images',metavar='TENSOR_SHAPE',default=[img_rows,img_cols,img_channels])
    parser.add_argument('-r','--rank',dest='rank',type=int,required=False,
                        help='rank of cutmix tile matrix for input images',metavar='RANK',default=tileRank)
    parser.add_argument('--mask', action='store_true',
                        help='Whether to enable masks on input images')
    parser.add_argument('--no-mask', action='store_false',
                        help='Whether to disable masks on input images')
    parser.add_argument('--prescale',dest='prescale',type=float,required=False,
                        default=prescale,help='prescale images by factor',metavar='PRESCALE')
    parser.set_defaults(mask=False)


def addTrainingArguments(parser) :

    augmentData=True
    learningRate=0.05
    maxNumEpochs=500
    snapshotCycle=60
    batchSize=30
    earlyStopPatience=-1
    parser.add_argument('-s','--split',dest='split',type=int,required=True,help='split for this training',metavar='SPLIT')
    parser.add_argument('--augment', action='store_true',
                        help='Whether to enable augmentation during training')
    parser.add_argument('--no-augment', action='store_false',
                        help='Whether to disable augmentation during training')
    parser.set_defaults(augment=augmentData)
    parser.add_argument('-e','--epochs',dest='epochs',type=int,required=False,default=maxNumEpochs,
                        help='Maximum number of epochs to train',metavar='EPOCHS')
    parser.add_argument('-c','--cycle',dest='cycle',type=int,required=False,default=snapshotCycle,
                        help='Cycle time for cosine-annealing when training',metavar='CYCLE')
    parser.add_argument('-l','--lr',dest='lr',type=float,required=False,default=learningRate,
                        help='Learning rate used for training',metavar='LR')
    parser.add_argument('-p','--patience',dest='patience',type=int,required=False,default=earlyStopPatience,
                        help='Early stop patience when training',metavar='PATIENCE')


def processCommandLine() :

    batchSize=30
    parser = argparse.ArgumentParser(description='Train Neural Net for some split.')
    addTrainingArguments(parser)
    addCommonArguments(parser,batchSize)
    args=parser.parse_args()
    if args.patience == -1 :
        args.patience=int(args.cycle*1.25)
    print('Training on Split: ' + str(args.split))
    print(f'Configuration Arguments:\n{args}\n')
    
    return args

