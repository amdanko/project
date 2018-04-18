
## MDSC 689.03 Project


### Organization


* **config** - contains examples of generated configuration files
* **nb** - contains jupyter notebooks :
   * **segmentation.ipynb** - predicted segmentations overlaid on top of the images, as well as providing the Dice score between the 'true' label and the predicted label
   * **stat_summary.ipynb** - analysis of training output
* **src** - modules found here 
* **results** - contain example log files (.npy and .hdf5 files are not included!)

If you wish to use this organization, I also suggest making a **data/** folder as well as a **sbatch** folder (or whatever you would like to call the folder you keep your .sh scripts in). Within that folder I also suggest keeping an **out/** folder, if your job scheduler tends to save your stdout 


### Scripts 

Everything is run using the ```main.py``` file, which can handle patch generation, model training, as well as producing predicted output. Each step must be run seperately and each step requires yaml files:

```
$ python main.py --newpatch --patch patch.yaml --model model.yaml 
$ python main.py --newmodel--patch patch.yaml --model model.yaml
$ python main.py --newpredict --patch patch.yaml --model model.yaml --testimg img.npy
```

Note about main: you are free to use the same yaml file for both flags

I also provide some example scripts that were useful during my workflow:

* ```make_config.py``` : I ran this first to generate all the configurations I was interested in 
* ```make_sbatch.py```: Used this to generate all the shell scripts, and then it prints out the command to run each one so I can just copy and paste everything into the terminal :)
* ```generate_results.py```: This gatherered all the separate log (.txt) files generated with each model, to make it easier to load up and analyze




