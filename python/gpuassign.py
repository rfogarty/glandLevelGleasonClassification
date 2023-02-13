import tensorflow as tf
from keras import backend as K

def assignNumCPUs(nthreads) :
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=nthreads,
                             inter_op_parallelism_threads=nthreads,
                             allow_soft_placement=True,
                             device_count={'CPU': nthreads})
    session = tf.compat.v1.Session(config=config)
    K.set_session(session)


def assignGPUs(gpuindices) :
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only use the first GPU
      try:
        visibleGPUs=[]
        for idx in gpuindices : visibleGPUs.append(gpus[idx])
        tf.config.set_visible_devices(visibleGPUs, 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
      except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

def gpuIds2Names(gpuindices) :
    # Return the list of GPU names that may be assigned to
    # distributed strategies
    GPUNames=[]
    for gpuNdx in gpuindices :
        GPUNames.append(f'GPU:{gpuNdx}')
    return GPUNames


def getStrategy(gpuNames) :
    if isinstance(gpuNames,str) : return tf.distribute.OneDeviceStrategy(device=gpuNames)
    elif len(gpuNames) == 1 : return tf.distribute.OneDeviceStrategy(device=gpuNames[0])
    else : return tf.distribute.MirroredStrategy(gpuNames)


if __name__ == "__main__" :
    # Running from commandline
    assignGPUs([0,1,2])

