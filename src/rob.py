import numpy as np 


def rgb_images(img,period):
    """
    This function creates a 4D with 3 channel ("RGB") image from a time-resolved periodic image
    Input:
    img -> gray-scale 3D image (time,H,W)
    period -> number of time points in a period
    Output:
    img_rgb -> RGB image
    """                                   
    t,H,W = img.shape
    img_rgb = np.zeros((t,H,W,3))
    for ii in xrange(t//period):
      #Previous time point
      img_rgb[ii*period+1:(ii+1)*period,:,:,0] = img[ii*period:(ii+1)*period-1,:,:]
      #Thre previous time point for the first is the last 
      img_rgb[ii*period,:,:,0] = img[(ii+1)*period-1,:,:]
      # Current time point 
      img_rgb[ii*period:(ii+1)*period,:,:,1] = img[ii*period:(ii+1)*period,:,:]
      # Next time point 
      img_rgb[ii*period:(ii+1)*period-1,:,:,2] =  img[ii*period+1:(ii+1)*period,:,:]	
      # The next time point of the last is the first
      img_rgb[(ii+1)*period-1,:,:,2] =  img[ii*period,:,:]	
    return img_rgb
    
    
    
def pad_images(samples,nmaxpooling = 4):
    """
    This function pads an image so its dimensions are a multiple of 2**nmaxpooling 
    Input:
    samples -> 4D samples array (nsamples,W,Z,nchannels)
    nmaxpooling -> Number of maxpooling layers in the CNN
    Output:
    samples_padded -> 4D padded samples array	(nsamples,W+nw,Z+nz,nchannels)
    nw -> padding amount (channel 1)
    nz -> padding amount (channel 2)
    """	
    nsamples,W,Z,nchannels = samples.shape
    nw = nmaxpooling**2-W%(2**nmaxpooling)
    nz = nmaxpooling**2-Z%(2**nmaxpooling)
    if nw == 0 and nz ==0:
        return samples,0,0
    samples_padded = np.concatenate((samples,np.zeros((nsamples,W,nz,nchannels))),axis = 2)
    samples_padded = np.concatenate((samples_padded,np.zeros((nsamples,nw,Z+nz,nchannels))),axis = 1)
    return samples_padded,nw,nz                                
 