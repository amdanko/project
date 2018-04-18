import os
import numpy as np

config_dir = np.array(os.listdir("config/"))

# i changed my mind ... prob dont need robunet d00 b/c that was  part of the point of his model..
config_dir = [v for i, v in enumerate(config_dir) if "rob_unet_d00" not in v]

# belatedly realized that n 5 or 10 makes no sense for p256.. 
config_dir = np.array(config_dir)
p256=[v for i, v in enumerate(config_dir) if "p256" in v]
p256=[v for i, v in enumerate(p256) if "n10" in v]
#config_dir = [v for i, v in enumerate(config_dir) if "p256" and "n10" not in v]
config_dir = [v for i, v in enumerate(config_dir) if v not in p256]


#for i in range(1,19):
for config in config_dir:
    job_name= "sbatch/"+config[:-5]+".sh"

    file = open(job_name, "w")
    
    file.write("#!/bin/bash \n")
    
    file.write("#SBATCH --account=def-rfrayne \n")
    #file.write("#SBATCH --chdir=/home/amdanko/projects/def-rfrayne/amdanko/research/seg/project \n")
    file.write("#SBATCH --gres=gpu:2\n")
    file.write("#SBATCH --nodes=1\n")
    file.write("#SBATCH --mem=16000M\n")
    file.write("#SBATCH --time=0-03:00\n")
    file.write("#SBATCH --job-name "+config[:-5]+"\n")
    file.write("#SBATCH --output sbatch/out/%x.out\n")
    file.write("#SBATCH --error sbatch/out/%x.err\n")

    file.write("'echo job started on:'\n")
    file.write("date\n")


    file.write("source /home/amdanko/projects/def-rfrayne/amdanko/car/bin/activate\n")
    
    file.write("python main.py --newpatch --patch config/"+config+" --model config/"+config+" \n")
    file.write("python main.py --newmodel --patch config/"+config+" --model config/"+config+" \n")
    file.write("python main.py --newpredict --patch config/"+config+" --model config/"+config+" --testimg snl_test.npy \n")

    file.write("'echo job ended on:'\n")
    file.write("date\n")
    file.close()
    
    

b_dir = np.array(os.listdir("sbatch/"))

# i changed my mind ... prob dont need robunet d00 b/c that was  part of the point of his model..
b_dir = [v for i, v in enumerate(b_dir) if "out" not in v]

#for i in range(1,19):
for b in b_dir:
    print "sbatch sbatch/"+b
