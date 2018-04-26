from __future__ import division
import os
import sys
import time
import random
import math
import numpy as np
import scipy
import yaml
import scipy
import nibabel as nib

from rob import rgb_images
from skimage.transform import rescale
from sklearn.feature_extraction.image import extract_patches_2d


#################





def load_seg_data(img_path, label_path,leaveOut=None, save=1,period=16):

    '''
    Load image and labels for patch generation! 

    Accepts two cases for img_path and label_path:
    
    - both .npy files 
        -- *** must be in the order: n,h,w ! ***

    - both are direcotries containing nifti files
        -- *** must be in separate directories ! ***
        -- ** all img must be same size  **
        -- ** labels and imgs must be in same alphabetical order (for now lol)  **'''

    # if both are .npy files
    
    if ".npy" in img_path and ".npy" in label_path:
        print ".npy files provided, loading those..."
        imgs = np.load(img_path)
        labels =np.load(label_path)
        
      
    # if both are directories
        
    elif os.path.isdir(img_path) and os.path.isdir(label_path):
        print "directories provided, loading nifti images..."
        img_files = np.array(os.listdir(img_path))
        label_files = np.array(os.listdir(label_path))
    
              
        # kick the one for cross validation out
        img_files=[v for i, v in enumerate(img_files) if leaveOut not in v]
        label_files=[v for i, v in enumerate(label_files) if leaveOut not in v]
        
        
        # make sure there is nothing sneaky in there 
        assert len(img_files)==len(label_files), "You have an unequal amount of images and labels!"
        
        test=[1 for i, v in enumerate(img_files) if "nii" in v]
        assert len(img_files) == sum(test), "there are non-nifti files in your image folder"
            
        test=[1 for i, v in enumerate(label_files) if "nii" in v]
        assert len(label_files) == sum(test), "there are non-nifti files in your label folder"
        del test 
        
        # all my files are 256x252.. 
        # so i don't feel like making this part
        # generalizeable ..lol
        imgs = np.zeros((1,256,252,3))
        labels = np.zeros((1,256,252,1))

        
        # go thru the images and actually load em in
        
        for (ii,jj) in zip(sorted(img_files),sorted(label_files)):
            aux_imgs = nib.load(os.path.join(img_path,ii)).get_data()
            aux_labels = nib.load(os.path.join(label_path,jj)).get_data()#[:,:,:,np.newaxis]

            # this is only because there are 2 imgs
            # in my data that are rotated 
            if aux_imgs.shape[0] == 252: 
                aux_imgs= np.swapaxes(aux_imgs,0,1)
            if aux_labels.shape[0] == 252:
                aux_labels= np.swapaxes(aux_labels,0,1)
        
            # tranpose 'em into the form we want: n,h,w
            aux_imgs=np.transpose(aux_imgs, [2,0,1]) 
            aux_labels=np.transpose(aux_labels, [2,0,1])

            #create 'time-rgb' images (rgb_images funct from post-doc)
            aux_imgs= rgb_images(aux_imgs,period)
            aux_labels=aux_labels[:,:,:,np.newaxis]
            imgs = np.concatenate((imgs,aux_imgs),axis = 0)
            labels = np.concatenate((labels,aux_labels),axis = 0)


        # beacuse we initialized it up there
        imgs = imgs[1:,:,:,:] 
        labels = labels[1:,:,:,:]   

        #Save the npy file for future use..
        # maybe set out_path = img_path by default ? for this 
        if (save):
            timestamp = str(int(time.time()))
            np.save(os.path.join(img_path,"no_"+leaveOut+"_img.npy"), imgs)
            np.save(os.path.join(label_path,"no_"+leaveOut+"_label.npy"), labels)
    
    else: 
        print "Improper image + label provided"
        print "Please provide either a directory containing nifti images + labels"
        print "or the path to a npy file containing the images + labels"
    
    return imgs,labels


def make_patch(img, label, out_path,period=16, patch=64, npatch=10, scale=1, norm=True, rgb=True, name="fart_town"):
    
    '''
    Generate new patches for training! 
    
    Extracts 'npatch' 2d patches per time-slice, for all images. Images are required to be 
    non shuffled to prevent my own confusion ... 
    
    Each patch corresponds to a randomly extracted portion of the image. Since we are 
    working on a segmentation task, we only save patches where part of the label is present 

   *** note: set to period=16 by default because all my data is period=16 ...     '''


    assert sum(sum(img[1,:,:,1] - img[0,:,:,2]))==0, """The data loaded is already shuffled, 
    but non-shuffled data is required. \n you can use load_seg_data() to get the data in the  right form"""

   
    if patch>=256:
        npatch=1 # dont eff it  ! 
    scans = img.shape[0]//period # determines length of loop
    samples=npatch*period*scans # determines 0th element of final array
    
    psize=float(patch)*scale # determines final patch size 
    psize=(int(psize),int(psize))

    
    # here is the final resting place of our patches: 
    img_patch=np.zeros(shape=(samples,psize[0],psize[1],3))
    label_patch=np.zeros(shape=(samples,psize[0],psize[1],1))

    
    if patch ==256: 
        img= np.pad(img,[[0,0],[0,0],[2,2],[0,0]],mode='edge')
        label = np.pad(label,[[0,0],[0,0],[2,2],[0,0]],mode='edge')
        
   
        
    for i in range(scans):
 
        # from the img and label arrays, extract portion corresponding to single img/scan/idk 
        start =i*period
        imgpoo= img[start:start+period,:,:,:]
        labelpoo= label[start:start+period,:,:,:]
        
        if (norm):
            img_max=imgpoo.max()
            img_min=imgpoo.min()
            imgpoo=100*(imgpoo-img_min)//(img_max-img_min)

            
        # start by extracting npatch # of slices


        for t in range(period):
            rs =random.randint(0,9999)
            
            if patch>=256:  # cant get a patc for something bgger than the img
                hm1=imgpoo[t,:,:,:]
                hm2=labelpoo[t,:,:,:np.newaxis]
                
            
            else:
                hm1= extract_patches_2d(imgpoo[t,:,:,:], patch_size=(patch,patch),random_state=rs,max_patches=npatch)
                hm2= extract_patches_2d(labelpoo[t,:,:,:],patch_size=(patch,patch),random_state=rs,max_patches=npatch)

                # array stating whether the patch has labels
                labelbool = np.zeros(npatch,dtype=bool) 

                for j  in range(npatch):
                    #labelbool=np.append(labelbool,1 in hm2[j,:,:])
                    labelbool[j]= 1 in hm2[j,:,:]
            
                # if we weren't able to extract at least 'npatch' patches containing labels,
                # continue to extract patches until we do
        
                while sum(labelbool) < npatch:
                    rs =random.randint(0,9999)
                    hm12= extract_patches_2d(imgpoo[1,:,:,:], patch_size=(patch,patch),random_state=rs,max_patches=1)
                    hm22= extract_patches_2d(labelpoo[1,:,:,:],patch_size=(patch,patch),random_state=rs,max_patches=1)
                    hm1=np.append(hm1,hm12,axis=0)
                    hm2=np.append(hm2,hm22,axis=0)
                    labelbool=np.append(labelbool,1 in hm22) #was hm12[-1,:,:,0] ?!

                #only keep the patche with labels    
                hm1=hm1[labelbool,:,:,:] #only keep if true
                hm2=hm2[labelbool,:,:,np.newaxis]
                #allocate to the finale array, scaling the patch if needed 
            
            if patch>=256:  #omg ... anyways just removed 'k' from everywhere ..npo
                hm1=imgpoo[t,:,:,:]
                hm2=labelpoo[t,:,:,:np.newaxis]
                if (scale!=1):
                    img_patch[(npatch*period*i)+(npatch*t):,:,:] =  rescale(
                        hm1[:,:,:],scale=scale,mode="constant")
                    label_patch[(npatch*period*i)+(npatch*t),:,:,:] = rescale(
                        hm2[:,:,:],scale=scale,mode="constant")
                else:
                    img_patch[(npatch*period*i)+(npatch*t),:,:,:] = hm1[:,:,:]
                    label_patch[(npatch*period*i)+(npatch*t),:,:,:] = hm2[:,:,:]

                              
            else:    
                for k in range(npatch):
                    if (scale!=1):
                        img_patch[(npatch*period*i)+(npatch*t)+k,:,:,:] =  rescale(
                            hm1[k,:,:,:],scale=scale,mode="constant")
                        label_patch[(npatch*period*i)+(npatch*t)+k,:,:,:] = rescale(
                            hm2[k,:,:,:],scale=scale,mode="constant")
                    else:
                        img_patch[(npatch*period*i)+(npatch*t)+k,:,:,:] = hm1[k,:,:,:]
                        label_patch[(npatch*period*i)+(npatch*t)+k,:,:,:] = hm2[k,:,:,:]




    np.save(os.path.join(out_path,name+"_img.npy"), img_patch)
    np.save(os.path.join(out_path,name+"_label.npy"), label_patch)
   
        
    return  





def load_patch(patch_path,patch_name,shuffle=1, val_perc=0.2, resize=1,save=1):

    '''
    Loads and returns training and validaion sets. 

    This takes in path for  the npy files.
    
    You have the following options:
        - Shuffle the data pror to separating into groups
        - Changing the % of data to be used for validation
        - Convert patch data to float32
        - '''

    print "Loading imgs..."
    img=np.load(os.path.join(patch_path,patch_name+"_img.npy"))
    label=np.load(os.path.join(patch_path,patch_name+"_label.npy"))

    if (resize):
        print "Resizing imgs to float32"
        img=img.astype(np.float32)
        label=label.astype(np.float32)

    if (shuffle):
        print "Shuffling imgs..."
        indexes = np.arange(img.shape[0],dtype = np.int32)
        random.shuffle(indexes)
        img = img[indexes,:,:,:]
        label = label[indexes,:,:,:]

        
    print "Separating into training and validation sets... "    
    # get number of validation imgs and separate them                  
    val_num = int(np.floor(img.shape[0]*val_perc))

    train_img = img[:-val_num]
    train_label = label[:-val_num]

    val_img = img[-val_num:]
    val_label = label[-val_num:]

    # feature scaling
    mean = np.mean(train_img)
    std = np.std(train_img)

    train_img -= mean
    train_img /= std

    val_img -= mean
    val_img /= std
                  
    if (save):
        print "Saving mean and std..."
        np.save(os.path.join(patch_path,patch_name+"_meanstd.npy"),np.array([mean,std]))
    
                 
    return train_img,train_label,val_img,val_label
                          

    

def load_predict_data(img_path, label_patch, leaveOut=None,save=1,period=16):
    '''
    Generate predicted patches and compare with ground truth '''
    img_files = np.array(os.listdir(img_path))
    label_files = np.array(os.listdir(label_path))


    # load ones originally kicked out 
    img_files=[v for i, v in enumerate(img_files) if leaveOut in v]
    label_files=[v for i, v in enumerate(label_files) if leaveOut in v]

    #next two temp
    img_files=[v for i, v in enumerate(img_files) if ".npy" not in v]
    label_files=[v for i, v in enumerate(label_files) if ".npy" not in v]

    img_names = [img.split(".nii.gz")[0] for img in img_files]

    # make sure there is nothing sneaky in there
    assert len(img_files)==len(label_files), "You have an unequal amount of images and labels!"


    imgs = np.zeros((1,256,252,3))
    labels = np.zeros((1,256,252,1))


    # go thru the images and actually load em in

    for (ii,jj) in zip(sorted(img_files),sorted(label_files)):
        aux_imgs = nib.load(os.path.join(img_path,ii)).get_data()
        aux_labels = nib.load(os.path.join(label_path,jj)).get_data()#[:,:,:,np.newaxis]

        # this is only because there are 2 imgs
        # in my data that are rotated
        if aux_imgs.shape[0] == 252:
            aux_imgs= np.swapaxes(aux_imgs,0,1)
        if aux_labels.shape[0] == 252:
            aux_labels= np.swapaxes(aux_labels,0,1)

        # tranpose 'em into the form we want: n,h,w
        aux_imgs=np.transpose(aux_imgs, [2,0,1])
        aux_labels=np.transpose(aux_labels, [2,0,1])

        #create 'time-rgb' images (rgb_images funct from post-doc)
        aux_imgs= rgb_images(aux_imgs,period)
        aux_labels=aux_labels[:,:,:,np.newaxis]

        imgs = np.concatenate((imgs,aux_imgs),axis = 0)
        labels = np.concatenate((labels,aux_labels),axis = 0)


    # beacuse we initialized it up there
    imgs = imgs[1:,:,:,:]
    labels = labels[1:,:,:,:]

    #Save the npy file for future use..
    # maybe set out_path = img_path by default ? for this
    if (save):
        timestamp = str(int(time.time()))
        np.save(os.path.join(img_path,leaveOut+"_img.npy"), imgs)
        np.save(os.path.join(label_path,leaveOut+"_label.npy"), labels)


    return imgs,labels, img_names
