import yaml
import os


template = {
    'file':{
        'img': 'data/orig/no_SNL_img.npy',
        'label': 'data/orig/no_SNL_label.npy',
        'leaveOut': 'SNL',
        'out': 'data/patch/',
        'name': 'TO_BE_DETERMINED'},
    'patch':{
        'size': 1234567890,
        'num': 1234567890,
        'scale': 1234567890,
        'norm': True,
        'rgb': True},
    'model':{
         'name': 'TO_BE_DETERMINED',
         'path': 'TO_BE_DETERMINED',
         'architecture': 'TO_BE_DETERMINED',
         'dropout': 'TO_BE_DETERMINED',
         'monitor': 'val_dice_coef',
         'epoch': 100,
         'early_stopping': True,
         'patience': 15},
    'augument':{
        'rotation_range': 15,
        'width_shift_range': 0.05,
        'height_shift_range': 0.05,
        'shear_range': 0.15,
        'zoom_range': 0.015,
        'horizontal_flip': True,
        'fill_mode': 'constant',
        'cval':0,
        'data_format': 'channels_last'}
}



patch_size=[64,128,256]
patch_scale=[1.0,0.50,0.25]
patch_num=[5,10]

model_arch = ['vanilla_unet','rob_unet','dil_unet']
model_path=['result/vanilla_unet','result/rob_unet','result/dil_unet']
model_dropout=[0,0.25]

for p in model_path: 
    if not os.path.exists(p):
        os.makedirs(p)

# generate patch names on the fly
# patch name format: 
# p64_s100_n5  : 64x64 patches, scaled 100%, 5 patches extracted/slice
# p128_s050_n10: 128x128 patches, scaled 50% (finalssize 64x64), 10 patches extracted/slice


#model_name will be generated on the fly -- iterate over outpath
# model name format:
# vanilla_unet_d00_p64_s100_n5 : vanilla unet with dropout 0.00 on patches same as above
# rob_unet_d25_p128_s050_n10: rob unet with dropout 0.25 on patches same as above 

for ma in model_arch: # use ma for model_path also
    for md in model_dropout:
        for pz in patch_size:
            for ps in patch_scale:
                for pn in patch_num:

                    if (pz==64) and (ps!=1.0):
                        continue 
                    if (pz==128) and (ps==0.25):
                        continue 
                        
                        
                    if (md==0):
                        drop="00"
                    if (md==0.25):
                        drop="25"

                    single_test = template

                    pname = "p"+str(pz)+"_s"+str(int(ps*100)).zfill(3)+"_n"+str(pn)
                    mname =  ma+"_d"+drop+"_"+pname

                  

                    single_test['file']['name'] = pname 
                    single_test['patch']['size'] = pz 
                    single_test['patch']['num'] = pn
                    single_test['patch']['scale'] = ps
                    single_test['model']['name'] = mname
                    single_test['model']['path'] = 'result/'+ma
                    single_test['model']['architecture'] = ma
                    single_test['model']['dropout'] = md

                    conf_file = "config/"+mname+".yaml"

                    #print single_test
                    
                    with open(conf_file, 'w') as outfile:
                        yaml.dump(single_test,outfile,default_flow_style=False)
                    
 


