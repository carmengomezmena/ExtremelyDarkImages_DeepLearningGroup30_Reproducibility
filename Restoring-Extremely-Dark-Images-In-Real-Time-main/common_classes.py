import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import rawpy
import glob
import imageio

def define_weights(num):                    # define initial weights
    weights = np.float32((np.logspace(0,num,127, endpoint=True, base=10.0))) # why is it 127 instead of 128 = 2^7?
                                            # logspace returns list of len 127 with numbers between 10^0=1 and 10^num.
    weights = weights/np.max(weights)       # divide by 10^num which makes every entry betwoon 0 and 1
    weights = np.flipud(weights).copy()     # This function reverses/flips the order of the list (so starting at 1 and slowly going down to 0)
    return weights                          # logspace list from 1 to 10^(-num)

    #normalised logspace between 1 (10^0) and given number (10^-num)

def get_na(bins,weights,img_loww,amp=1.0):
    #Function: Define pre-amplification coefficients
    #input: bins(normalised log space base 2 (0-1)), weights (normalised log space base 10 (0-10^-num) and img_loww (resized raw image)
    #output: Numerical Aperture of image (numerical number = range of angles over which a system can accept of emit light
    H,W = img_loww.shape
    arr = img_loww*1
    selection_dict = {weights[0]: (bins[0]<=arr)&(arr<bins[1])}
    for ii in range(1,len(weights)):
        selection_dict[weights[ii]] = (bins[ii]<=arr)&(arr<bins[ii+1])
    mask = np.select(condlist=selection_dict.values(), choicelist=selection_dict.keys())
   
    mask_sum1 = np.sum(mask,dtype=np.float64)
    
    na1 = np.float32(np.float64(mask_sum1*0.01*amp)/np.sum(img_loww*mask,dtype=np.float64))

    if na1>300.0: #bound na <=300
        na1 = np.float32(300.0)
    if na1<1.0: #bound na >=1
        na1 = np.float32(1.0)
    
    selection_dict.clear()

    return na1


def part_init(train_files):
    #Function: Create list of amplified images from raw imput images
    #Input: Training files/Raw input files
    #Output: list of training amplified images

    bins = np.float32((np.logspace(0,8,128, endpoint=True, base=2.0)-1))/255.0
    #creates bins: normalised logspace base 2 -> 2^8 = 256 so 256-1/255 = 1 (so logspace between 0-1)
    weights5 = define_weights(5) #normalised logspace between 0 and 5 in inverse order? -> why 5?
    train_list = [] #empty list to be filled
    
    for i in range(len(train_files)):
        
        raw = rawpy.imread(train_files[i]) #read raw file
        img = raw.raw_image_visible.astype(np.float32).copy() #make img from raw file -> array of floats
        raw.close() #close raw file (we have img now)
        
        h,w = img.shape #get height and width of file
        if h%32!=0: #if height not correct dimentions, reshape
            print('Image dimensions should be multiple of 32. Correcting the 1st dimension.')
            h = (h//32)*32
            img = img[:h,:]
        
        if w%32!=0: #if width not correct dimentions, reshape
            print('Image dimensions should be multiple of 32. Correcting the 2nd dimension.')
            w = (w//32)*32
            img = img[:,:w]        

        img_loww = (np.maximum(img - 512, 0)/ (16383 - 512))    #img_loww = maximum between (img-512) and 0, divided by (16383 - 512)
        # img_loww = (np.maximum(img - 2**9, 0)/((2**14)-1)-2**9)
        # 512 = 16x32,  16384 (!) = 32x512  (why is it 16383?)
        na5 = get_na(bins,weights5,img_loww)  #apply get_na with the defined bins, weights and image -> get 'numerical aperture' factor
        
        img_loww = img_loww*na5 #multiply image by na factor
            
        train_list.append(img_loww) #amplified image append to list

        print('Image No.: {}, Amplification_m=1: {}'.format(i+1,na5))
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

        
