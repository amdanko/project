#!/usr/bin/env python

import yaml
import os



'''
Defines image and model ojbects, mainly their properties

'''




class training_img(object):
    def __init__(self,img_yaml):

        self.yaml = os.path.normpath(img_yaml)

        self.load_yaml()

    def load_yaml(self):
        self.img_options=[]
        with open(self.yaml) as ymlfile:
            self.img_options = yaml.load(ymlfile)
             #        return img_options
            # file/dir options

    def get_file_info(self):
        return self.img_options['file']


    def get_patch_info(self):
        return self.img_options['patch']


    @property
    def img_path(self):
        return self.img_options['file']['img']

    @property
    def label_path(self):
        return self.img_options['file']['label']

    @property
    def leaveOut(self):
        return self.img_options['file']['leaveOut']

    @property
    def out_path(self):
        return self.img_options['file']['out']

    @property
    def out_name(self):
        return self.img_options['file']['name']

        # patch options
    @property
    def patch_size(self):
        return self.img_options['patch']['size']

    @property
    def patch_n(self):
        return self.img_options['patch']['num']

    @property
    def patch_scale(self):
        return self.img_options['patch']['scale']

    @property
    def patch_norm(self):
        return self.img_options['patch']['norm']

    @property
    def patch_rgb(self):
        return self.img_options['patch']['rgb']


class training_opt(object):
    def __init__(self,run_yaml):

        self.yaml = os.path.normpath(run_yaml)

        self.load_yaml()

    def load_yaml(self):
        self.img_options=[]
        with open(self.yaml) as ymlfile:
            self.run_options = yaml.load(ymlfile)
             #        return img_options
            # file/dir options

    def get_data_aug(self):
        return self.run_options['augument']


    def get_model_info(self):
        return self.run_options['model']


    @property
    def model_name(self):
        return self.run_options['model']['name']

    @property
    def model_path(self):
        return self.run_options['model']['path']

    @property
    def model_arch(self):
        return self.run_options['model']['architecture']

    @property
    def model_monitor(self):
        return self.run_options['model']['monitor']

    @property
    def model_epoch(self):
        return self.run_options['model']['epoch']

    @property
    def model_dropout(self):
        return self.run_options['model']['dropout']

    @property
    def early_stopping(self):
        return self.run_options['model']['early_stopping']

    @property
    def model_patience(self):
        return self.run_options['model']['patience']

    @property
    def rotation_range(self):
        return self.run_options['augument']['rotation_range']

        # patch options
    @property
    def width_shift_range(self):
        return self.run_options['augument']['width_shift_range']

    @property
    def height_shift_range(self):
        return self.run_options['augument']['heigh_shift_range']

    @property
    def shear_range(self):
        return self.run_options['augument']['shear_range']

    @property
    def zoom_range(self):
        return self.run_options['augument']['zoom_range']

    @property
    def horizontal_flip(self):
        return self.run_options['augument']['horizontal_flip']

    @property
    def fill_mode(self):
        return self.run_options['augument']['fill_mode']

    @property
    def cval(self):
        return self.run_options['augument']['cval']
