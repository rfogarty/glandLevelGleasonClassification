
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense,Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
from keras.layers import BatchNormalization
from gpuassign import *
import numpy as np


def assembleNet(input_tensor_shape,numClasses,augment=True,seed=17,dropout=0.65) :
    
    inputModel = layers.Input(input_tensor_shape)
    headModel = inputModel
    if augment :
        headModel = layers.experimental.preprocessing.RandomFlip(seed=seed)(headModel)
    
    base_model = VGG16(weights='imagenet', include_top=False)
    for layer in base_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable  = False
    
    headModel = base_model(headModel)
    headModel = GlobalAveragePooling2D()(headModel)
    headModel = Flatten()(headModel)  # This seems to be added implicitly - RBF
    headModel = BatchNormalization()(headModel)
    if dropout >= 0.0 : headModel = Dropout(dropout)(headModel)
    headModel = Dense(32, activation='relu',
                      kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
                      bias_regularizer=regularizers.l2(1e-4))(headModel)
    headModel = BatchNormalization()(headModel)
    if dropout >= 0.0 : headModel = Dropout(dropout)(headModel)
    if numClasses > 2 :
        headModel = Dense(numClasses, activation='softmax')(headModel)
        model = Model(inputs=inputModel, outputs=headModel)
        print(model.summary())
    elif numClasses == 2 :
        headModel = Dense(2, activation='sigmoid')(headModel)
        model = Model(inputs=inputModel, outputs=headModel)
        print(model.summary())
    else :
        print(f'FATAL: numClasses({numClasses}) < 2')
        model = None
    
    return (model,base_model)


def compileNet(model,numClasses) :
    # Set optimization strategy (maybe parameterize this someday.
    opt = SGD(momentum=0.9)
    # And compile multi or binary classification models
    if numClasses > 2 :
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
    elif numClasses == 2 :
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=[tf.keras.metrics.BinaryAccuracy(),
                               tf.keras.metrics.Recall(),
                               tf.keras.metrics.TruePositives(),
                               tf.keras.metrics.TrueNegatives(),
                               tf.keras.metrics.FalsePositives(),
                               tf.keras.metrics.FalseNegatives(),
                               tf.keras.metrics.AUC(num_thresholds=20)])


def buildTrainingNet(input_tensor_shape,numClasses,gpuNames,augment=True,seed=17) :
    
    strategy = getStrategy(gpuNames)

    with strategy.scope():
        model,base_model = assembleNet(input_tensor_shape,numClasses,augment=augment,seed=seed)
        if model != None :
            compileNet(model,numClasses)
            
    return model


def buildFineTuningNet(modelfile,input_tensor_shape,numClasses,gpuNames,augment=True,seed=17,trainFeatureLayers=True) :
    
    strategy = getStrategy(gpuNames)

    with strategy.scope():
        model,base_model = assembleNet(input_tensor_shape,numClasses,augment=augment,seed=seed)
        if model != None :
            # Now load model weights
            model.load_weights(modelfile)

            # Now reenable training on VGG16 layers
            if trainFeatureLayers :
                for layer in base_model.layers:
                    if not isinstance(layer, BatchNormalization):
                        layer.trainable  = True
            
            compileNet(model,numClasses)

    return model


def buildInferenceNet(input_tensor_shape,numClasses,gpuNames) :
    strategy = getStrategy(gpuNames)
    with strategy.scope():
        model,base_model = assembleNet(input_tensor_shape,numClasses,augment=False,dropout=0.0)
    
    return model


def createEnsembleModel(modelfiles,input_tensor_shape,numClasses,args) :
    modelEnsemble = [0] * len(modelfiles)
    for idx,filename in enumerate(modelfiles) :
        print(f'INFO: adding {filename} to ensemble')
        modelEnsemble[idx] = buildInferenceNet(input_tensor_shape,numClasses,gpuIds2Names(args.gpus))
        modelEnsemble[idx].build(input_shape = (1,input_tensor_shape[0],input_tensor_shape[1],input_tensor_shape[2]))
        modelEnsemble[idx].load_weights(filename)
    return modelEnsemble


def createFeatureModel(modelfile,input_tensor_shape,numClasses,args) :
    print(f'INFO: Creating model for {modelfile}')
    model = buildInferenceNet(input_tensor_shape,numClasses,gpuIds2Names(args.gpus))
    model.build(input_shape = (1,input_tensor_shape[0],input_tensor_shape[1],input_tensor_shape[2]))
    model.load_weights(modelfile)
    # Now replace model with just the features:
    model = Model(inputs=model.input,outputs=model.get_layer(name='cnnFeatures').output)
    return model


def createModel(modelfile,input_tensor_shape,numClasses,args) :
    model = buildInferenceNet(input_tensor_shape,numClasses,gpuIds2Names(args.gpus))
    model.build(input_shape = (1,input_tensor_shape[0],input_tensor_shape[1],input_tensor_shape[2]))
    model.load_weights(modelfile)
    return model


def makeFeatures(testDataIterator,modelfile,input_tensor_shape,numClasses,args) :
    
    model = createFeatureModel(modelfile,input_tensor_shape,numClasses,args)

    print(f"INFO: computing features for {modelfile}")
    features = model.predict(testDataIterator)
    print(f"INFO: features.shape {features.shape}")
    
    return features


def ensemblePredict(model,testData,numClasses) :
    predictions = model.predict(testData)
    maxed = np.argmax(predictions, axis=1)
    # This is a super clever way of creating onehot vectors
    # note in our multinomial kaggle case, the 4 labels are: GS0, GS3, GS4, GS5
    # Credit: https://stackoverflow.com/questions/49684379/numpy-equivalent-to-keras-function-utils-to-categorical
    onehots=np.eye(numClasses)[maxed]
    return onehots


def makeMultinomialEnsembleInferencePredictions(testDataIterator,modelfiles,input_tensor_shape,numClasses,args,findTrueConsensus=True) :
    models = createEnsembleModel(modelfiles,input_tensor_shape,numClasses,args)
    
    if findTrueConsensus :
        # Or I can vote which would be a better approach
        print(f"INFO: computing predictions for {len(modelfiles)} ensemble")
        predictions = np.array([ensemblePredict(model,testDataIterator) for model in models])
        print(f"INFO: predictions.shape {predictions.shape}")
        return predictions
    else :
        # We can either sum these up:
        print(f"INFO: computing predictions for {len(modelfiles)} ensemble")
        predictions = np.array([model.predict(testDataIterator) for model in models])
        print(f"INFO: predictions.shape {predictions.shape}")
        return predictions


def makeBinomialEnsembleInferencePredictions(testDataIterator,modelfiles,input_tensor_shape,numClasses,args,findTrueConsensus=True) :
    models = createEnsembleModel(modelfiles,input_tensor_shape,numClasses,args)
    print(f"INFO: computing predictions for {len(modelfiles)} ensemble")
    predictions = np.array([model.predict(testDataIterator) for model in models])
    print(f"INFO: predictions.shape {predictions.shape}")
    outcomes=(predictions > 0.5).astype(int)
    return (outcomes,predictions)


def makeMultinomialEnsembleInferences(testDataIterator,modelfiles,input_tensor_shape,numClasses,args,findTrueConsensus=True) :
    predictions=makeMultinomialInferencePredictions(testDataIterator,modelfiles,input_tensor_shape,numClasses,args,findTrueConsensus=True)
    outcomes = np.sum(predictions, axis=0)
    return outcomes


def makeBinomialInferences(testDataIterator,modelfile,input_tensor_shape,numClasses,args,findTrueConsensus=True) :
    # Note, this is an ensemble of 1
    outcomes,predictions=makeBinomialEnsembleInferencePredictions(testDataIterator,[modelfile],input_tensor_shape,numClasses,args,findTrueConsensus=findTrueConsensus)
    outcomes = np.sum(outcomes, axis=0)
    return (outcomes,predictions)


################################################################################################
# This code leveraged information and source code by Jason Brownlee on setting up ensemble networks.
# https://machinelearningmastery.com/model-averaging-ensemble-for-deep-learning-neural-networks/
def makeEnsembleInferences(testDataIterator,modelfiles,input_tensor_shape,numClasses,args) :
    
    models = createEnsembleModel(modelfiles,input_tensor_shape,numClasses,args)

    print(f"INFO: computing predictions for {len(modelfiles)} ensemble")
    predictions = np.array([model.predict(testDataIterator) for model in models])
    print(f"INFO: predictions.shape {predictions.shape}")
    
    # sum across ensembles
    outcomes = np.sum((predictions > 0.5).astype(int), axis=0)
    return (outcomes,predictions)


def makeMultinomialInferences(testDataIterator,modelfile,input_tensor_shape,numClasses,args) :
    
    model = createModel(modelfile,input_tensor_shape,numClasses,args)

    print(f"INFO: computing predictions for {modelfile}")
    predictions = model.predict(testDataIterator)
    print(f"INFO: predictions.shape {predictions.shape}")

    # argmax across classes
    outcomes = np.argmax(predictions, axis=1) 
    return (outcomes,predictions)
