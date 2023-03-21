import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import rawpy
import glob
import imageio
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
def define_weights(num):                    # define initial weights
    # weights = np.float32((np.logspace(0,num,127, endpoint=True, base=10.0))) # why is it 127 instead of 128 = 2^7?
    weights = np.float32((np.logspace(0, num, 9, endpoint=True, base=10.0)))  # why is it 127 instead of 128 = 2^7?
                                            # logspace returns list of len 127 with numbers between 10^0=1 and 10^num.
    weights = weights/np.max(weights)       # divide by 10^num which makes every entry between 0 and 1
    weights = np.flipud(weights).copy()     # This function reverses/flips the order of the list (so starting at 1 and slowly going down to 0)
    return weights                          # logspace list from 1 to 10^(-num)

    #normalised logspace between 1 (10^0) and given number (10^-num)

def get_na(bins,weights,img_loww,amp=1.0):
    #Function: Define amplification factor of image based on weighted averages of pixels intensity
    #input: bins (normalised log space base 2 (0-1)), weights (normalised log space base 10 (0-10^-num) and img_loww (2D array resized/normalised/bias corrected raw image)
    #output: Numerical Amplification factor for the image
    H,W = img_loww.shape #this is not used?
    arr = img_loww*1 #this is making a copy of img_loww, why not img_loww.copy()?
    selection_dict = {weights[0]: (bins[0]<=arr) & (arr<bins[1])} # if the pixel value (arr) is between the first bin value and the second bin value, the weight = weight[0] = 1
    for ii in range(1,len(weights)):
        selection_dict[weights[ii]] = (bins[ii]<=arr) & (arr<bins[ii+1])
    mask = np.select(condlist=selection_dict.values(), choicelist=selection_dict.keys()) # Select: Return an array drawn from elements in choicelist, depending on conditions.
   
    mask_sum1 = np.sum(mask,dtype=np.float64)
    
    na1 = np.float32(np.float64(mask_sum1*0.01*amp)/np.sum(img_loww*mask,dtype=np.float64))

    if na1>300.0: #bound na <=300
        na1 = np.float32(300.0)
    if na1<1.0: #bound na >=1
        na1 = np.float32(1.0)
    
    selection_dict.clear()
    print('na1: ', na1)
    return na1


""" Example:
weights = [5., 4., 3., 2., 1., 0.]
bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
arr = np.array([0.15, 0.33, 0.45, 0.01, 0.03])

selection_dict = {weights[0]: (bins[0]<=arr) & (arr<bins[1])} # if the pixel value (arr) is between the first bin value and the second bin value, the weight = weight[0] = 1
for ii in range(1, len(weights)):
    selection_dict[weights[ii]] = (bins[ii]<=arr) & (arr<bins[ii+1])
print(selection_dict)
 """

def part_init(train_files):
    #Function: Create list of amplified images from raw imput images
    #Input: Training files/Raw input files
    #Output: list of training amplified images

    # 128 bins, the first bin = 0 -> bins used to quantify intensity range
    #bins = np.float32((np.logspace(0,8,128, endpoint=True, base=2.0)-1))/255.0
    bins = np.float32((np.logspace(0, 8, 10, endpoint=True, base=2.0) - 1)) / 255.0
    #creates bins: normalised logspace base 2 -> 2^8 = 256 so 256-1/255 = 1 (so logspace between 0-1)
    #128 points define 127 bins
    # Equation 6 in Paper - Normalization is -> bk = 2**(k*8/n) / 2**8 (mathiness)

    weights5 = define_weights(5) #normalised logspace between 0 and 5 in inverse order? -> why 5?
    train_list = [] #empty list to be filled

    plt.figure()
    # plt.xscale("log")
    plt.title('Bins & Weights')
    print('bins: ', bins)
    for b in bins:
            plt.axvline(x=b, color = 'red', ls = '--')
    plt.scatter(np.arange(1,10), weights5, color = 'green')

    for i in range(len(train_files)):
        
        raw = rawpy.imread(train_files[i]) #read raw file, create 'raw' class
        img = raw.raw_image_visible.astype(np.float32).copy() #make img from raw file -> array of floats
        #For Bayer images, a 2D ndarray is returned.
        #For Foveon and other RGB - type images, a 3D ndarray is returned.

        raw.close() #close raw file (we have img now)
        print('shape is: ',img.shape)

        h,w = img.shape #get height and width of file = (h = 2848, w = 4256)
        if h%32!=0: #if height not correct dimentions, reshape
            print('Image dimensions should be multiple of 32. Correcting the 1st dimension.')
            h = (h//32)*32
            img = img[:h,:]
        
        if w%32!=0: #if width not correct dimentions, reshape
            print('Image dimensions should be multiple of 32. Correcting the 2nd dimension.')
            w = (w//32)*32
            img = img[:,:w]        

        # print('Img after resize is: ', img)
        # print('Img max is: ', max(img[0]), max(img[1]))
        # img is a 2D array (2848, 4256) with 'one value per pixel' - darkest values are around the value 512

        img_loww = (np.maximum(img - 512, 0)/ (16383 - 512))    #img_loww = maximum between (img-512) and 0, divided by (16383 - 512)
        # img_loww = (np.maximum(img - 2**9, 0)/((2**14)-1)-2**9)
        # 512 = 16x32,  16384 (!) = 32x512  (why is it 16383?)
        #numbers below 512 are turned to 0 (considered black), numbers above 512 are normalised to lie between 0 and 1

        #512 could be bias for non-zero black level (so correcting for the fact that the camera does not capture black properly) usually due to noise in sensor
        # This bias might be camera specific...
        #16383 is maximum pixel value for Bayer image -> bit depth for 14-bit Bayer image (value range (2**0-1) - (2**14-1))

        # Paper says 8-bit sensors? So why is here the max pixel size for 14-bit
        # Is this just making the image darker? Why if we wouldn't take the noise away?

        # print('Img_loww is: ', img_loww)

        na5 = get_na(bins,weights5,img_loww)  #apply get_na with the defined bins, weights and image -> get 'numerical amplification' factor
        # label_na = str('NA for pic: '+ str(i))
        # plt.axhline(na5, color = 'blue', label = label_na)
        img_loww = img_loww*na5 #multiply image by na factor (every pixel)
        # print('after amplification: ', img_loww)
        train_list.append(img_loww) #amplified image append to list

        print('Image No.: {}, Amplification_m=1: {}'.format(i+1,na5)) #m=1 is hardcoded into get_na function
    plt.legend()
    return train_list
    
    
################ DATASET CLASS
class load_data(Dataset):  #creates class for loading data
    """Loads the Data."""
    
    def __init__(self, train_files):
        # class initiation by calling 'part_init' function and assigning the outcome (train_list) to self.train_list
        print('\n...... Loading all files to CPU RAM\n')
        self.train_list = part_init(train_files)        
        print('\nFiles loaded to CPU RAM......\n')
        
    def __len__(self): #function to get length of training list
        return len(self.train_list)

    def __getitem__(self, idx):  #function to get a certain item (at index idx) from the training list
        img_low = self.train_list[idx]
        return torch.from_numpy(img_low).float().unsqueeze(0) #item is a float and 'unsqueezed'

def run_test(model, dataloader_test, save_images):
    # Function: Creates and saves 3 prediction images from each dark data image inputted
    #inputs: model, dataloader_test, save_images
    #output: /
    with torch.no_grad(): #disable gradient calculation in pytorch (used when backward() not used, so less calculations)
        model.eval() #evaluate model
        for image_num, low in enumerate(dataloader_test):
            low = low.to(next(model.parameters()).device) #save model parameters in 'low'
            for amp in [1.0,5.0,8.0]: #three amplification factors -> three predictions per image
                pred = model(amp*low) #prediction = run model for amplification*parameter
                pred = (np.clip(pred[0].detach().cpu().numpy().transpose(1,2,0),0,1)*255).astype(np.uint8) #modify prediction
                imageio.imwrite(os.path.join(save_images,'img_num_{}_m_{}.jpg'.format(image_num,amp)), pred) #save prediction
    return

        
