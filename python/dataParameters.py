import os
from sklearn.preprocessing import LabelBinarizer
import numpy as np

def dataDir() :
    # Need to provide path to directory that folds file listing
    #return '/data/CombinedKaggleUMMCC'
    return '/data/MoffittPathology/All/Regions'

def masksDir() :
    return None

def maskExt() :
    return None

def tensorSize() :
    img_rows, img_cols, img_channels = 300,300,3
    return (img_rows,img_cols,img_channels)

def checkpointMonitor() :
    monitor='val_categorical_accuracy'
    mode='max'
    return (monitor,mode)

def earlyStopMonitor() :
    monitor='val_loss'
    mode='min'
    return (monitor,mode)

##################################################################
# Data sets
def trainingSet(split) :
    return 'training_set.list'

def validationSet(split) :
    return f'validation_set{split}.list'

def validationSetAlt1(split) :
    return f'validation_set{split}_ummcc.list'

def validationSetAlt2(split) :
    return f'validation_set{split}_panda.list'

def validationSetAlt3(split) :
    return f'gs3_and_gs4.list'

def tuningSet(split) :
    return trainingSet(split)

# To be used for tests like RANSACing
def seenTestSet(split) :
    return trainingSet(split)

def holdoutTestSet(split) :
    return None

##################################################################
# Blocklists
def trainingBlocklists(split) :
    data_dir=dataDir()
    # Currently, this is set up to block nothing.
    blocklists=None
    # Here are blocklists when performing GS3 vs GS4 tests
    #blocklists=[f'{data_dir}/Kaggle/gs0-5_blocklist.txt',f'{data_dir}/validation_set{split}.list']
    return blocklists

def validationBlocklists(split) :
    data_dir=dataDir()
    blocklists=None
    #blocklists=[f'{data_dir}/Kaggle/gs0-5_blocklist.txt']
    return blocklists

def validationBlocklistsAlt1(split) :
    data_dir=dataDir()
    blocklists=None
    return blocklists

def validationBlocklistsAlt2(split) :
    return validationBlocklists(split)

def validationBlocklistsAlt3(split) :
    data_dir=dataDir()
    blocklists=None
    #blocklists=[f'{data_dir}/blocklistTooSmall.txt',f'{data_dir}/blocklistTooLarge.txt']
    return blocklists

def tuningBlocklists(split) :
    blocklists=trainingBlocklists(split)
    blocklists=None
    #blocklists.append('blocklist.txt')
    return blocklists

def seenTestBlocklists(split) :
    return validationBlocklists(split)

def holdoutBlocklists(split) :
    return None

##################################################################
#  History files
def trainingHistory(split) :
    return f'trainHistoryDict-split{split}.pickle'

def finetuneHistory(split) :
    return f'finetuneHistoryDict-split{split}.pickle'

##################################################################
#  Path to Label converters
def numClasses() :
    return 4

def classLabels() :
    classes = ['GS3-PANDA','GS3-UMMCC','GS4-PANDA','GS4-UMMCC']
    # Process through LabelBinarizer to ensure consistency with reporting tools
    lb = LabelBinarizer()
    lb.fit(classes)
    return (lb.classes_)

def labels2indices(labels) :
    # Process through LabelBinarizer to ensure consistency with reporting tools
    lb = LabelBinarizer()
    lb.fit(classLabels())
    return lb.transform(labels)
 

# If labels are aggregated into smaller set of labels, that should be done here
def relabel(classes) :
    #breakpoint()
    if len(classes.shape) == 2 :
        if classes.shape[1] == 1 :
            classes = np.reshape(classes,classes.shape[0])
        elif classes.shape[1] > 1 :
            classes = np.argmax(classes,axis=1)
    classes[classes == 1] = 0
    classes[classes == 2] = 1
    classes[classes == 3] = 1
    return classes

def binarizePredictions(predictions) :
    # Converting to single sigmoid from softmax probabilities
    # Note: all predictions add to 1 (softmax condition)
    # So a sum across all would yield a maximum of 1
    # If equally distributed across all classes, then we'd want a sigmoid output ~0.5.
    # If skewed far to negative class we want ~0.0, and skewed far to positive class, we want ~1.0.
    predictions_n = -np.sum(predictions[:,0:2],axis=1)
    predictions_p = np.sum(predictions[:,2:4],axis=1)
    predictions_c = np.sum((predictions_n,predictions_p),axis=0)
    predictions_c = predictions_c + 1
    predictions_c = predictions_c / 2
    return predictions_c

def path2label(imagePath) : 
    label = imagePath.split(os.path.sep)[-2]
    savedlabel = label
    # Modified the below to test independent Moffitt test set
    if label == 'GS3' :
        if 'stack' in imagePath or 'Layer' in imagePath :
            label = 'GS3-UMMCC'
        else :
            label = 'GS3-PANDA'
    else : # GS4
        if 'stack' in imagePath or 'Layer' in imagePath :
            label = 'GS4-UMMCC'
        else :
            label = 'GS4-PANDA'
    return label

def path2subject(imagePath) :
    basepath = os.path.basename(imagePath)
    ## UM/MCC specific (all UM/MCC data has "stack" in the name due to the .vsi image format)
    #subject = basepath.split('s')[0] # s for "stack"
    ## PANDA Radboud specific
    #subject = basepath.split('_')[0]
    ## PANDA or UM/MCC
    if basepath != None and 'stack' in basepath :
        subject = basepath.split('s')[0] # s for "stack"
    elif basepath != None and 'Layer' in basepath :
        subject = basepath.split('_')[0]
    elif basepath != None:
        subject = basepath.split('_')[0]
    else :
        subject = None
    return subject


