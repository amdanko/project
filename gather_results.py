import os
import numpy as np
import yaml
import pandas as pd

import argparse

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result",type=str,default="default", help="supply the folder with logs")
    args=parser.parse_args()
    return args

# assumes txt or yaml 



def default_dir():
    print "loading vanilla,rob,dil results"
    vanilla_dir = np.array(os.listdir("result/vanilla_unet"))
    rob_dir = np.array(os.listdir("result/rob_unet"))
    dil_dir = np.array(os.listdir("result/dil_unet"))

    vanilla_dir= [v for i , v in enumerate(vanilla_dir) if ".txt" in v]
    rob_dir= [v for i , v in enumerate(rob_dir) if ".txt" in v]
    dil_dir= [v for i , v in enumerate(dil_dir) if ".txt" in v]
    vanilla_dir = ["result/vanilla_unet/" + x for x in vanilla_dir]
    rob_dir = ["result/rob_unet/" + x for x in rob_dir]
    dil_dir = ["result/dil_unet/" + x for x in dil_dir]


    combined_dir = vanilla_dir+rob_dir+dil_dir
    
    return combined_dir

def requested_dir(folder):
    print "loading "
    result_dir = np.array(os.listdir(folder))
    result_dir = [v for i, v in enumerate(result_dir) if ".txt" in v]
    prepend = folder if folder[-1:]=="/" else folder+"/"
    result_dir = (prepend + x for x in result_dir)
    return result_dir  
    

if __name__ == "__main__":
    
    args=parseArgs()    
    
    result_list = requested_dir(args.result) if args.result!="default" else default_dir()
    
    for result in result_list: 
        with open(result,'r') as f:
            df = pd.io.json.json_normalize(yaml.load(f))
        if result == result_list[0]:
            result_df = df
        else: 
            result_df = result_df.append(df)
            
            
    #result_df.index = range(len(result_df))
    
    result_df.to_csv(path_or_buf="result/results.txt",index=False)