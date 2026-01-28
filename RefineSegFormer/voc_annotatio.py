import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm


trainval_percent    = 1
train_percent       = 0.9

VOCdevkit_path      = 'VOCdevkit'

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    saveBasePath    = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')
    
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num     = len(total_seg)  
    list    = range(num)  
    tv      = int(num*trainval_percent)  
    tr      = int(tv*train_percent)  
    trainval= random.sample(list,tv)  
    train   = random.sample(trainval,tr)  
    
    print("train and val size",tv)
    print("traub suze",tr)
    ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    for i in list:  
        name = total_seg[i][:-4]+'\n'  
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
    print("Generate txt in ImageSets done.")

    print("Check datasets format, this may take a while.")
    classes_nums        = np.zeros([256], int)
    for i in tqdm(list):
        name            = total_seg[i]
        png_file_name   = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("Label picture%ss not detected. Please check whether the file exists in the specific path and whether the suffix is PNG."%(png_file_name))
        
        png             = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("The shape of the label image%s is%s, which is not a grayscale image or an eight bit color image. Please carefully check the data set format."%(name, str(np.shape(png))))
            print("The label image needs to be a grayscale image or an eight bit color image. The value of each pixel of the label is the type of the pixel."%(name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
            
    print("Print the value and number of pixels.")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)
    
    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("It is detected that the value of pixels in the label only contains 0 and 255, and the data format is incorrect.")
        print("For the binary classification problem, the label needs to be modified to have a background pixel value of 0 and a target pixel value of 1.")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("It is measured that the label only contains background pixels, and the data format is incorrect. Please carefully check the data set format.")

    print("The image in jpegimages should be a.Tiff or Tif file, and the image in segmentationclass should be a.Png file.")