import os
import yaml
import src
import argparse
import numpy as np

from src.dataset import training_img,training_opt
from src.data_loader import load_seg_data, make_patch,load_patch
from src.data_augumentation import aug_gen

from src.keras_krap import vanilla_unet,rob_unet,dil_unet,train_seg, seg_img



#
# Welcome to ... the main frame?
# This is the main program, can be used to make patches, train model, and predict a segmentation -- please only do one at a time
# This is explained in further detail elsewhere
#

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--newpatch",action="store_true",default=False, help="make new patches for training (do this prior to training, cannot be run together)")
    parser.add_argument("--newmodel",action="store_true",default=False, help="make new model")
    parser.add_argument("--newpredict",action="store_true",default=False, help="make new predicted images")
    parser.add_argument("--patch",type=str,default="patch.yaml", help="supply the patch yaml file")
    parser.add_argument("--model",type=str,default="model.yaml", help="supply the model yaml file ")
    parser.add_argument("--testimg",type=str,default="something.npy", help="supply image to test ")
    args=parser.parse_args()
    return args


def new_patch(img_obj):
    ''' Generate patches to use in future training runs'''

    # load  data
    img, label = load_seg_data(img_path=img_obj.img_path, label_path=img_obj.label_path,
                               leaveOut=img_obj.leaveOut)
    # make data
    make_patch(img=img, label=label, out_path=img_obj.out_path, name=img_obj.out_name,
               patch=img_obj.patch_size, npatch=img_obj.patch_n, scale=img_obj.patch_scale,
               norm=img_obj.patch_norm,rgb=img_obj.patch_rgb)
    return


def train_model(img_obj,train_obj,log=1):
    ''' Train a model for the configuration provided'''

    # get training imgs, labels and val imgs, labels
    train_img,train_label,val_img,val_label = load_patch(patch_path=img_obj.out_path, patch_name=img_obj.out_name)


    # get arguments for the data augumentation generator
    gen_arg = train_obj.get_data_aug()

    # create the data augumentation generator .. bs is batch size
    combined = aug_gen(train_img,train_label,bs=32,**gen_arg)

    psize= img_obj.patch_size*img_obj.patch_scale
    if train_obj.model_arch=="vanilla_unet":
        model = vanilla_unet(patch_size=(psize,psize),dropout=train_obj.model_dropout)
    elif train_obj.model_arch=="rob_unet":
        model = rob_unet(patch_size=(psize,psize),dropout=train_obj.model_dropout)
    elif train_obj.model_arch=="dil_unet":
        model=dil_unet(patch_size=(psize,psize),dropout=train_obj.model_dropout)
    else:
        print "you didn't chose one of the possible model architecutres: "
        print "- vanilla_unet \n -rob_unet \n dil_unet"

    # this looks more scary than it really is , I made the property names same as
    # the arugument names for a reason!
    run_dict = train_seg(train_patch=train_img, train_label=train_label,
                      val_patch=val_img, val_label=val_label, img_gen=combined,
                      model=model, model_path=train_obj.model_path, model_name= train_obj.model_name,
                      epochs=train_obj.model_epoch, monitor=train_obj.model_monitor,
                      early_stop=train_obj.early_stopping, patience=train_obj.model_patience)

    print run_dict

    if (log):
        make_log(img_obj,train_obj,**run_dict)

    return



def make_log(img_obj,train_obj,**run_dict):
    # makes the .txt file which summarizes everything

    file  = img_obj.get_file_info()
    patch = img_obj.get_patch_info()
    model = train_obj.get_model_info()
    aug = train_obj.get_data_aug()
    log_dict = {'file': file,
               'patch': patch,
               'model': model,
               'augument': aug,
               'run': run_dict}
    log_file = os.path.join(train_obj.model_path,train_obj.model_name+".txt")
    # gonna print to the same output as the model ..
    with open(log_file, 'w') as outfile:
        yaml.dump(log_dict, outfile, default_flow_style=False)

    print log_dict
    return

<<<<<<< HEAD



def make_predict_OG(img_obj,train_obj,test_img):
=======
def make_predict(img_obj,train_obj,test_img):
    # just makes the predicted image
>>>>>>> master
    print "Loading trained model..."
    test_model=os.path.join(train_obj.model_path,train_obj.model_name+".hdf5")
    mean_t, std_t = np.load(os.path.join(img_obj.out_path,img_obj.out_name+"_meanstd.npy"))

    if train_obj.model_arch=="vanilla_unet":
        model = vanilla_unet()
        model.load_weights(test_model)
    elif train_obj.model_arch=="rob_unet":
        model = rob_unet()
        model.load_weights(test_model)
    elif train_obj.model_arch=="dil_unet":
        model=dil_unet()
        model.load_weights(test_model)
    else:
        print "you didn't chose one of the possible model architecutres: "
        print "- vanilla_unet \n -rob_unet \n dil_unet"

    print "Loading test image..."

<<<<<<< HEAD
    img, label = load_predict_data(img_path=img_obj.img_path, label_path=img_obj.label_path,
                               leaveOut=img_obj.leaveOut)
        
=======

    img = np.load(test_img)
>>>>>>> master
    print "Trying to segment..."
    predict = seg_img(model,img,mean_t,std_t)
    #predicted_imgs.append(predict)
    print "Saving Segmentation Result..."

    np.save(os.path.join(train_obj.model_path,train_obj.model_name+".npy"),predict)

    return


def make_predict(img_obj,train_obj,test_img):

    #LOAD TESTING IMG 
    print "Loading test image..."

    img, label,names = load_predict_data(img_path=img_obj.img_path, label_path=img_obj.label_path,
                               leaveOut=img_obj.leaveOut)
        
    # THEN LOAD MODEL 
    
    print "Loading trained model..."
    test_model=os.path.join(train_obj.model_path,train_obj.model_name+".hdf5")
    mean_t, std_t = np.load(os.path.join(img_obj.out_path,img_obj.out_name+"_meanstd.npy"))
    
    if train_obj.model_arch=="vanilla_unet":
        model = vanilla_unet()
        model.load_weights(test_model)
    elif train_obj.model_arch=="rob_unet":
        model = rob_unet()
        model.load_weights(test_model)
    elif train_obj.model_arch=="dil_unet":
        model=dil_unet()
        model.load_weights(test_model)
    else:
        print "you didn't chose one of the possible model architecutres: "
        print "- vanilla_unet \n -rob_unet \n dil_unet"
        
    print "Trying to segment..."
        
    predict = seg_img(img,label,names,model,mean_t,std_t)    
    #predicted_imgs.append(predict)
    print "Saving Segmentation Result..."
    
    np.save(os.path.join(train_obj.model_path,train_obj.model_name+".npy"),predict)

    return




if __name__ == "__main__":

    args=parseArgs()

    if not (args.newpredict):
        assert bool(args.newpatch) ^ bool(args.newmodel), "Sorry, you can't make new patches and train new model for the same run! .. for now"

    # load in the yaml files
    patch_info=training_img(args.patch)
    model_info=training_opt(args.model)

    # make new patches if we want
    # but first: check if patches we want exist -- no need to remake them
    if (args.newpatch):
        if (os.path.exists(os.path.join(patch_info.img_path,patch_info.out_name+"_img.npy"))) and (
            os.path.exists(os.path.join(patch_info.img_path,patch_info.out_name+"_label.npy"))):
            print "Patches already exist, exiting:"
        else:
            new_patch(patch_info)

    # make new model if desired
    elif (args.newmodel):
        model_log = train_model(patch_info,model_info)

    # make make predicted segmentation    
    if (args.newpredict):
        make_predict(patch_info,model_info,args.testimg)



  #  else:
  #      print "you have made a grave mistake"
