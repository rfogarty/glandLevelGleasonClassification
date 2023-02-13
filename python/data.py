import os
import numpy as np
import random
import cv2
import re
import itertools
from dataParameters import *
# Versioning specific utils
import pkg_resources as pkgs
from packaging import version as pkg_version
from platform import python_version

tfversion = pkg_version.parse(pkgs.get_distribution('tensorflow').version)
if tfversion < pkg_version.parse('2.9') :
    from keras.preprocessing.image import img_to_array
else :
    from tensorflow.keras.utils import img_to_array

from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from tensorflow import keras


def parseBlocklistWithDatadir(filepath,prepend_path) :
    blocklist=[]
    with open(filepath) as f:
        lines = f.readlines()
    for line in lines :
        line = line.lstrip().rstrip()
        #if not line and not line.startswith('#') :
        if line and not line.startswith('#') :
            line = os.path.join(prepend_path,line)
            blocklist.append(line)
    return blocklist


def parseBlocklist(filepath,trimpath=False) :
    blocklist=[]
    with open(filepath) as f:
        lines = f.readlines()
    for line in lines :
        line = line.lstrip().rstrip()
        #if not line and not line.startswith('#') :
        if line and not line.startswith('#') :
            if trimpath: line = os.path.basename(line)
            blocklist.append(line)
    return blocklist


def readMask(impath) :
    mask=cv2.imread(impath)
    img_channels = len(mask.shape)
    if(img_channels == 3) : mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    mask = mask.astype('float32')
    mask = np.ceil(mask/np.amax(mask))
    return mask


def applyMask(image,mask) :
    img_channels = len(image.shape)
    if img_channels == 3 :
        image = image*mask[:,:,None]
    elif img_channels == 2 :
        image = image*mask[:,:]
    return image


def augmentHue(image,factor) :
    imageHSV=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    vectorHSV=imageHSV.reshape((image.shape[0]*image.shape[1],3))
    h=vectorHSV[:,0]
    if image.dtype == 'uint8' :
        intfactor=int(factor*255)
        randHueShift=np.random.randint(intfactor)-int(intfactor/2)
        h=h+randHueShift
        # Note: since dtype is uint8, values will naturally wrap around in the above step
    elif image.dtype == 'uint16' :
        intfactor=int(factor*65535)
        randHueShift=np.random.randint(intfactor)-int(intfactor/2)
        h=h+randHueShift
        # Note: since dtype is uint16, values will naturally wrap around in the above step
    else : # assume it is unit normalized
        intfactor=int(factor*255)
        randHueShift=np.random.randint(intfactor)-int(intfactor/2)
        randHueShift=randHueShift/255.0
        h=h+randHueShift
        if randHueShift > 0.0 :
            h[h>1.0] = h[h>1.0] - 1.0
        else :
            h[h<0.0] = h[h<0.0] + 1.0
    vectorHSV[:,0]=h
    imageHSV=vectorHSV.reshape((image.shape[0],image.shape[1],3))
    image=cv2.cvtColor(imageHSV,cv2.COLOR_HSV2RGB)
    return image


def loadImage(impath,img_channels=3,mask=None,prescale=1.0) :
    image = cv2.imread(impath)
    if (img_channels == 1) : image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    elif (img_channels == 3) : image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    if mask is not None :
        mask = readMask(mask)
        image = applyMask(image,mask)
    if 1.0 > prescale or prescale > 1.0 : # i.e. prescale != 1
        image = cv2.resize(image, (0,0), fx=prescale,fy=prescale)
    return image


def standardizeImage(image,normalize=True) :
    if normalize :
        image = image.astype('float32')
        m=np.mean(image)
        s=np.std(image)
        image = (image - m)/s
    
    return image


def readImage(impath,input_tensor_shape=(224,224,3),hueAugment=None,normalize=True,mask=None) :
    img_rows,img_cols,img_channels=input_tensor_shape

    image = loadImage(impath,img_channels,mask,prescale=1.0)
    image = cv2.resize(image, (img_rows, img_cols))
    
    if hueAugment is not None:
        image = augmentHue(image,hueAugment)

    image = standardizeImage(image,normalize)
    return image


def padImage(image,padded_w,padded_h):
    assert image.shape[1] <= padded_w
    assert image.shape[0] <= padded_h
    border_w = padded_w - image.shape[1]
    border_h = padded_h - image.shape[0]
    left = int(border_w/2)
    right = border_w - left
    top = int(border_h/2)
    bottom = border_h - top
    image = cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT)
    return image


def randomCrop(image,crop_w,crop_h):
    assert image.shape[1] >= crop_w
    assert image.shape[0] >= crop_h
    max_offset_x = image.shape[1] - crop_w
    max_offset_y = image.shape[0] - crop_h
    if max_offset_x > 0 : x = np.random.randint(0, max_offset_x)
    else : x = 0
    if max_offset_y > 0 : y = np.random.randint(0, max_offset_y)
    else : y = 0
    crop = image[y:(y + crop_h),x:(x + crop_w)]
    return crop


def samplemix(image,crop_w,crop_h,tilerank):
    orig_crop_w = crop_w
    orig_crop_h = crop_h
    orig_tilerank = tilerank
    # This is a hack to scale down the samplemix if the source image
    # is too small.
    while image.shape[1] < crop_w or image.shape[0] < crop_h :
       crop_w = int(crop_w / 2)
       crop_h = int(crop_h / 2)
       tilerank = tilerank * 2
    assert image.shape[1] >= crop_w, f'image width({image.shape[1]}) < crop_w ({crop_w})'
    assert image.shape[0] >= crop_h, f'image height({image.shape[0]}) < crop_h ({crop_h})'
    
    max_offset_x = image.shape[1] - crop_w
    max_offset_y = image.shape[0] - crop_h
    
    x_delta = max_offset_x / (tilerank-1)
    y_delta = max_offset_y / (tilerank-1)
    
    if image.ndim == 3 :
        imsamplemix = np.zeros((orig_crop_h*orig_tilerank,orig_crop_w*orig_tilerank,image.shape[2]),dtype=image.dtype)
    else : # lif image.ndim == 2 :
        imsamplemix = np.zeros((orig_crop_h*orig_tilerank,orig_crop_w*orig_tilerank),dtype=image.dtype)
    
    for tx in range(tilerank) :
        txoffdst = int(tx*crop_w)
        txoffsrc = int(tx*x_delta)
        for ty in range(tilerank) :
            tyoffdst = int(ty*crop_h)
            tyoffsrc = int(ty*y_delta)
            imsamplemix[tyoffdst:tyoffdst+crop_h,txoffdst:txoffdst+crop_w] = image[tyoffsrc:tyoffsrc+crop_h,
                                                                              txoffsrc:txoffsrc+crop_w]
    return imsamplemix


def readPaddedImage(impath,input_tensor_shape=(224,224,3),prescale=1.0,
                    hueAugment=None,normalize=True,mask=None) :
    img_rows,img_cols,img_channels=input_tensor_shape

    image = loadImage(impath,img_channels,mask,prescale)
    image = padImage(image,img_cols,img_rows)
    
    if hueAugment is not None:
        image = augmentHue(image,hueAugment)

    image = standardizeImage(image,normalize)
    return image


def readRandomCropOfImage(impath,input_tensor_shape=(224,224,3),prescale=1.0,
                          hueAugment=None,normalize=True,mask=None) :
    img_rows,img_cols,img_channels=input_tensor_shape

    image = loadImage(impath,img_channels,mask,prescale)
    image = randomCrop(image,img_cols,img_rows)
    
    if hueAugment is not None:
        image = augmentHue(image,hueAugment)

    image = standardizeImage(image,normalize)
    return image


def readSamplemixOfImage(impath,input_tensor_shape=(224,224,3),tilerank=3,prescale=1.0,
                      hueAugment=None,normalize=True,mask=None) :
    img_rows,img_cols,img_channels=input_tensor_shape

    image = loadImage(impath,img_channels,mask,prescale)
    image = samplemix(image,int(img_cols/tilerank),int(img_rows/tilerank),tilerank)
    if hueAugment is not None:
        image = augmentHue(image,hueAugment)

    image = standardizeImage(image,normalize)
    return image


def maskName(img_name,mask_dir,mask_ext=None) :
    mask_name=None
    if mask_dir is not None :
        if mask_ext is not None :
            prefix, ext = os.path.splitext(img_name)
            mask_name=prefix + mask_ext
        else : mask_name = img_name
        mask_name=os.path.join(mask_dir,mask_name)
    elif mask_ext is not None :
        prefix, ext = os.path.splitext(img_name)
        mask_name=prefix + mask_ext
    return mask_name


class ResizedGenerator :
    def __init__(self,image_dir,input_tensor_shape=(224,224,3),hueAugment=None,
                 normalize=True,mask_dir=None,mask_ext=None) :
        self.image_dir = image_dir
        self.input_tensor_shape = input_tensor_shape
        self.hueAugment = hueAugment
        self.normalize = normalize
        self.mask_dir = mask_dir
        self.mask_ext = mask_ext

    def __call__(self,img_name) :
        image=readImage(self.image_dir + img_name,self.input_tensor_shape,hueAugment=self.hueAugment,
                        normalize=self.normalize,mask=maskName(img_name,self.mask_dir,self.mask_ext))
        return image


class FramedGenerator :
    def __init__(self,image_dir,input_tensor_shape=(224,224,3),prescale=1.0,hueAugment=None,
                 normalize=True,mask_dir=None,mask_ext=None) :
        self.image_dir = image_dir
        self.input_tensor_shape = input_tensor_shape
        self.prescale = prescale
        self.hueAugment = hueAugment
        self.normalize = normalize
        self.mask_dir = mask_dir
        self.mask_ext = mask_ext

    def __call__(self,img_name) :
        image=readPaddedImage(self.image_dir + img_name,self.input_tensor_shape,
                              prescale=self.prescale,hueAugment=self.hueAugment,normalize=self.normalize,
                              mask=maskName(img_name,self.mask_dir,self.mask_ext))
        return image


class CroppedGenerator :
    def __init__(self,image_dir,input_tensor_shape=(224,224,3),tilerank=3,prescale=1.0,hueAugment=None,
                 normalize=True,mask_dir=None,mask_ext=None) :
        self.image_dir = image_dir
        self.input_tensor_shape = input_tensor_shape
        self.prescale = prescale
        self.hueAugment = hueAugment
        self.normalize = normalize
        self.mask_dir = mask_dir
        self.mask_ext = mask_ext

    def __call__(self,img_name) :
        image=readRandomCropOfImage(self.image_dir + img_name,self.input_tensor_shape,
                                    prescale=self.prescale,hueAugment=self.hueAugment,normalize=self.normalize,
                                    mask=maskName(img_name,self.mask_dir,self.mask_ext))
        return image


class SamplemixGenerator :
    def __init__(self,image_dir,input_tensor_shape=(224,224,3),tilerank=3,prescale=1.0,hueAugment=None,
                 normalize=True,mask_dir=None,mask_ext=None) :
        self.image_dir = image_dir
        self.input_tensor_shape = input_tensor_shape
        self.tilerank = tilerank
        self.prescale = prescale
        self.hueAugment = hueAugment
        self.normalize = normalize
        self.mask_dir = mask_dir
        self.mask_ext = mask_ext
        if 1.0 > self.prescale or self.prescale > 1.0 :
            print(f'INFO: Prescaling images by: {self.prescale}')

    def __call__(self,img_name) :
        image=readSamplemixOfImage(self.image_dir + img_name,self.input_tensor_shape,self.tilerank,
                                prescale=self.prescale,hueAugment=self.hueAugment,normalize=self.normalize,
                                mask=maskName(img_name,self.mask_dir,self.mask_ext))
        return image


def divisable(dividend,divisor) :
    return dividend % divisor == 0


def optimizeBalance(most_classes,fewest_classes,batch_size,num_classes=2,numPools=1) :
    if batch_size > fewest_classes*num_classes :
        print(f'WARNING: batch_size({batch_size}) too large for # of elements')
        batch_size = fewest_classes*num_classes
        print(f'WARNING: batch_size resized to {batch_size}')
    disprop_ratio=int(np.ceil(most_classes/fewest_classes))
    numBalancedImages=num_classes*most_classes
    saved_most_classes=most_classes
    batchIterations = (np.ceil( (numBalancedImages/float(numPools)) / float(batch_size))).astype(int)
    while disprop_ratio > numPools or not divisable(numBalancedImages,numPools*batch_size) :
        if disprop_ratio > numPools :
            numPools = disprop_ratio
            # The following prevents us from increasing both numPools
            # and most_classes in a single iteration which could lead to badness...
            most_classes=saved_most_classes
            numBalancedImages=num_classes*most_classes
        batchIterations = (np.ceil( (numBalancedImages/float(numPools)) / float(batch_size))).astype(int)
        numBalancedImages = batchIterations*numPools*batch_size
        # Note: paddedSize should be divisible by num classes
        most_classes = int(np.ceil(numBalancedImages/num_classes))
        numBalancedImages=num_classes*most_classes
        disprop_ratio=int(np.ceil(most_classes/fewest_classes))
    return (most_classes,numPools,batch_size,batchIterations)


def balancedDataset(image_files,labels,batch_size,numPools) : 
    separated_images={}
    # Need to do some preprocessing to separate the images into n lists.
    labelNums=np.argmax(labels,axis=1)
    for ndx in range(labels.shape[0]) :
        separated_images.setdefault(labelNums[ndx], []).append(image_files[ndx])
    num_classes = len(separated_images)
    fewest_classes=1e100;
    most_classes=0;
    # Check the min and max length for all classes
    for clist in separated_images.values() :
        if  most_classes < len(clist) :
            most_classes = len(clist)
        if  fewest_classes > len(clist) :
            fewest_classes = len(clist)
    
    paddedSize,numPools,batch_size,batchIterations = optimizeBalance(most_classes,fewest_classes,
                                                                     batch_size,num_classes,numPools)

    # Make sure everything is shuffled
    image_lists=[]
    image_lists.append(range(int(paddedSize)))
    label_lists=[]
    label_lists.append(range(int(paddedSize)))
    for k,v in separated_images.items() :
        print(f'INFO: Label({k}) has {len(v)} images')
        label_lists.append(itertools.repeat(k))
        random.shuffle(v)
        image_lists.append(itertools.cycle(v))

    images=[]
    for t in zip(*image_lists) :
        images.extend(t[1:])
    labels=[]
    for t in zip(*label_lists) :
        labels.extend(t[1:])
    labels = np.array(labels)
    labels.resize(len(labels),1)
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    return (images,labels,numPools,batch_size,batchIterations)


class AutoBalancingPooledDataAndLabel_Iterator(keras.utils.Sequence) :
    def __init__(self,generator,image_files,labels,batch_size,numPools=1,autoBalance=True,shuffle=False) :
        self.generator = generator
        if autoBalance :
           image_files,labels,numPools,batch_size,batchIterations = balancedDataset(image_files,labels,batch_size,numPools)
        self.images = image_files
        self.labels = labels
        self.batch_size = batch_size
        self.numPools = numPools
        self.pool = 0
        self.autoBalance = autoBalance
        # Given autobalancing, it is a little bit tricky to do true shuffling,
        # so we only support a pseudo shuffle, by shuffling the order of the batches
        # and we will also shuffle intra each batch.
        self.shuffle = shuffle
        
        # Wrap some images and labels so that we ensure we have enough data for each pool-batch if
        # not already wrapped by auto-balancing. A consequence is that the last pool and the first pool
        # will have a few of the same samples, which is perfectly fine.
        if not autoBalance :
            batchIterations = (np.ceil( (len(self.images)/float(self.numPools)) / float(self.batch_size))).astype(int)
            if self.numPools > 1 :
                paddedSize = batchIterations*self.numPools*self.batch_size
                wrappedImages = self.images+self.images[0:paddedSize-len(self.images)]
                self.images = wrappedImages
                wrappedLabels = np.concatenate((self.labels,self.labels[0:paddedSize-len(self.labels)]))
                self.labels = wrappedLabels
        
        self.batch_ndx = [i for i in range(batchIterations)]
    
    def getImages(self) :
        return self.images
    
    def getLabels(self) :
        return self.labels
    
    def on_epoch_end(self) :
        savedPool=self.pool
        self.pool = self.pool + 1
        if self.pool >= self.numPools :
            self.pool = 0
        if savedPool != self.pool :
            print(f'\nSwitching data to pool: {self.pool}\n')
        if self.shuffle:
            random.shuffle(self.batch_ndx)

    def __len__(self) :
        return (np.ceil( (len(self.images)/float(self.numPools)) / float(self.batch_size))).astype(int)
    
    def __getitem__(self, idx) :
        idx_warp = self.batch_ndx[idx]
        idxpool=int((self.pool*self.__len__())+idx_warp)

        imbatch = self.images[idxpool * self.batch_size : (idxpool+1) * self.batch_size]
        labbatch = self.labels[idxpool * self.batch_size : (idxpool+1) * self.batch_size]
        # If "pseudo" shuffling, besides shuffling the "batch" itself (with idx_warp above),
        # we also shuffle intra each batch.
        if self.shuffle :
            bdx = [i for i in range(len(imbatch))]
            random.shuffle(bdx)
            imbatch=np.array(imbatch)
            return (np.array([self.generator(img_name) for img_name in imbatch[bdx]]),labbatch[bdx])
        else :
            return (np.array([self.generator(img_name) for img_name in imbatch]),labbatch)


class Data_Iterator(keras.utils.Sequence) :
    def __init__(self, generator, image_files, batch_size) :
        self.generator = generator
        self.images = image_files
        self.batch_size = batch_size
      
    def __len__(self) :
        return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)
    
    def __getitem__(self, idx) :
        imbatch = self.images[idx * self.batch_size : (idx+1) * self.batch_size]
        return np.array([self.generator(img_name) for img_name in imbatch])


def readLabels(imagePaths,data_dir='') :
    labels=[]
    for imagePath in imagePaths:
        labels.append(path2label(imagePath))
    
    labels = np.array(labels)
    # Generate label class indices
    numeric_labels = labels2indices(labels)
    print(f'numeric_labels.shape: {numeric_labels.shape}')
    return (numeric_labels)


# This code loosely based on imutils library but restructured to allow
# for following along symbolic links and 4 cases were split out
# to allow for the highest efficiency when iterating a list of directories
# with the fewest checks and string processing possible.
def list_files(basePath, validExts=None, contains=None) :
    if ((contains is not None) and (validExts is not None)) :
        # loop over the directory structure
        for (baseDir, dirNames, filenames) in os.walk(basePath,followlinks=True) :
            # loop over the filenames in the current directory
            for filename in filenames :
                # if the contains string is not none and the filename does not contain
                # the supplied string, then ignore the file
                if filename.find(contains) == -1 : continue

                # determine the file extension of the current file
                ext = filename[filename.rfind("."):].lower()

                # check to see if the file has one of the extensions specified
                if ext.endswith(validExts) :
                    # construct the path to the image and yield it
                    filePath = os.path.join(baseDir, filename)
                    yield filePath

    elif ((contains is None) and (validExts is not None)) :
        # loop over the directory structure
        for (baseDir, dirNames, filenames) in os.walk(basePath,followlinks=True) :
            # loop over the filenames in the current directory
            for filename in filenames :
                # determine the file extension of the current file
                ext = filename[filename.rfind("."):].lower()

                # check to see if the file has one of the extensions specified
                if ext.endswith(validExts) :
                    # construct the path to the image and yield it
                    filePath = os.path.join(baseDir, filename)
                    yield filePath

    elif ((contains is not None) and (validExts is None)) :
        # loop over the directory structure
        for (baseDir, dirNames, filenames) in os.walk(basePath,followlinks=True) :
            # loop over the filenames in the current directory
            for filename in filenames :
                # if the contains string is not none and the filename does not contain
                # the supplied string, then ignore the file
                if filename.find(contains) == -1 : continue

                # construct the path to the image and yield it
                filePath = os.path.join(baseDir, filename)
                yield filePath

    else : # ((contains is None) and (validExts is None))
        # loop over the directory structure
        for (baseDir, dirNames, filenames) in os.walk(basePath,followlinks=True) :
            # loop over the filenames in the current directory
            for filename in filenames :
                # construct the path to the image and yield it
                filePath = os.path.join(baseDir, filename)
                yield filePath


def list_images(basePath,image_types=(".pgm",".ppm",".jpg",".jpeg",".png",".bmp",".tif",".tiff"),contains=None) :
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def filterFilelist(filelist,filterstr=None,blocklist=None) :
    if filterstr is not None :
        filelist = [p for p in filelist if re.search(filterstr,p)]
    if blocklist is not None :
        # first convert list to set
        fileset = set(filelist)
        for b in blocklist : 
            if b in fileset :
                fileset.remove(b)
        filelist = list(fileset)
    filelist.sort()
    return filelist


def processImageDir(data_dir,filterstr,blocklist,shuffled) :
    # grab the image paths and randomly shuffle them
    print(f"[INFO] loading images from {data_dir}...")
    imagePaths = sorted(list(list_images(data_dir)))
    if blocklist is not None :
        if type(blocklist) in [list,tuple] :
            blocked=[]
            for b in blocklist :
                blocked.extend(parseBlocklistWithDatadir(b,data_dir))
            blocklist=blocked
        else :
            blocklist = parseBlocklistWithDatadir(blocklist,data_dir)
    imagePaths = filterFilelist(imagePaths,filterstr,blocklist)
    if shuffled:
        random.seed(14)
        random.shuffle(imagePaths)
    return imagePaths


def processImageList(data_dir,file_list,filterstr,blocklist,shuffled) :
    # grab the image paths and randomly shuffle them
    print(f"[INFO] loading images from {data_dir}/{file_list}...")
    imagePaths = None
    with open(os.path.join(data_dir,file_list)) as f:
        lines = []
        for line in f.readlines() :
            linestripped=line.strip()
            if len(linestripped) > 0 :
                linepath = os.path.join(data_dir,linestripped)
                if os.path.exists(linepath) : # O.w. assume it is a comment or something else
                    lines.append(linepath)
        imagePaths=sorted(lines)
    if blocklist is not None :
        if type(blocklist) in [list,tuple] :
            blocked=[]
            for b in blocklist :
                blocked.extend(parseBlocklistWithDatadir(b,data_dir))
            blocklist=blocked
        else :
            blocklist = parseBlocklistWithDatadir(blocklist,data_dir)
    imagePaths = filterFilelist(imagePaths,filterstr,blocklist)
    if shuffled:
        random.seed(14)
        random.shuffle(imagePaths)
    return imagePaths


def getDataResizedIteratorFromListing(data_dir,file_list,filterstr=None,blocklist=None,shuffled=False,batch_size=32,numPools=1,autoBalance=True,
                                     input_tensor_shape=(224,224,3),hueAugment=None,normalize=True,mask_dir=None,mask_ext=None) :
    imagePaths = processImageList(data_dir,file_list,filterstr,blocklist,shuffled)
    labels = readLabels(imagePaths)
    generator = ResizedGenerator('',input_tensor_shape=input_tensor_shape,hueAugment=hueAugment,normalize=normalize,mask_dir=mask_dir,mask_ext=mask_ext)
    dataIterator = AutoBalancingPooledDataAndLabel_Iterator(generator, imagePaths, labels, batch_size, numPools,autoBalance=autoBalance,shuffle=shuffled)

    return (dataIterator,dataIterator.getLabels(),dataIterator.getImages())


def getDataFramedIteratorFromListing(data_dir,file_list,filterstr=None,blocklist=None,shuffled=False,batch_size=32,numPools=1,autoBalance=True,
                                     input_tensor_shape=(224,224,3),prescale=1.0,hueAugment=None,normalize=True,mask_dir=None,mask_ext=None) :
    imagePaths = processImageList(data_dir,file_list,filterstr,blocklist,shuffled)
    labels = readLabels(imagePaths)
    generator = FramedGenerator('',input_tensor_shape=input_tensor_shape,prescale=prescale,hueAugment=hueAugment,normalize=normalize,mask_dir=mask_dir,mask_ext=mask_ext)
    dataIterator = AutoBalancingPooledDataAndLabel_Iterator(generator, imagePaths, labels, batch_size, numPools,autoBalance=autoBalance,shuffle=shuffled)
    return (dataIterator,dataIterator.getLabels(),dataIterator.getImages())


def getDataCropIteratorFromListing(data_dir,file_list,filterstr=None,blocklist=None,shuffled=False,batch_size=32,numPools=1,autoBalance=True,
                                     input_tensor_shape=(224,224,3),prescale=1.0,hueAugment=None,normalize=True,mask_dir=None,mask_ext=None) :
    imagePaths = processImageList(data_dir,file_list,filterstr,blocklist,shuffled)
    labels = readLabels(imagePaths)
    generator = CroppedGenerator('',input_tensor_shape=input_tensor_shape,prescale=prescale,hueAugment=hueAugment,normalize=normalize,mask_dir=mask_dir,mask_ext=mask_ext)
    dataIterator = AutoBalancingPooledDataAndLabel_Iterator(generator, imagePaths, labels, batch_size, numPools,autoBalance=autoBalance,shuffle=shuffled)

    return (dataIterator,dataIterator.getLabels(),dataIterator.getImages())


def getDataSamplemixIteratorFromListing(data_dir,file_list,filterstr=None,blocklist=None,shuffled=False,batch_size=32,numPools=1,autoBalance=True,
                                     input_tensor_shape=(224,224,3),tilerank=3,prescale=1.0,hueAugment=None,normalize=True,mask_dir=None,mask_ext=None) :
    imagePaths = processImageList(data_dir,file_list,filterstr,blocklist,shuffled)
    labels = readLabels(imagePaths)
    generator = SamplemixGenerator('',input_tensor_shape=input_tensor_shape,tilerank=tilerank,prescale=prescale,hueAugment=hueAugment,normalize=normalize,mask_dir=mask_dir,mask_ext=mask_ext)
    dataIterator = AutoBalancingPooledDataAndLabel_Iterator(generator, imagePaths, labels, batch_size, numPools,autoBalance=autoBalance,shuffle=shuffled)
    return (dataIterator,dataIterator.getLabels(),dataIterator.getImages())


def getDataSamplemixIteratorFromDir(data_dir,filterstr=None,blocklist=None,shuffled=False,batch_size=30,numPools=1,autoBalance=True,
                                 input_tensor_shape=(224,224,3),tilerank=3,prescale=1.0,hueAugment=None,normalize=True,mask_dir=None,mask_ext=None) :
    imagePaths = processImageDir(data_dir,filterstr,blocklist,shuffled)
    labels = readLabels(imagePaths)
    generator = SamplemixGenerator('',input_tensor_shape=input_tensor_shape,tilerank=tilerank,prescale=prescale,hueAugment=hueAugment,normalize=normalize,mask_dir=mask_dir,mask_ext=mask_ext)
    dataIterator = AutoBalancingPooledDataAndLabel_Iterator(generator, imagePaths, labels, batch_size, numPools,autoBalance=autoBalance,shuffle=shuffled)
    return (dataIterator,dataIterator.getLabels(),dataIterator.getImages())


