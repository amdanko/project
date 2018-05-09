
import keras
from keras.preprocessing.image import ImageDataGenerator



'''
This creates data augmentation generator, so my data can be continuously transformed during the training process


'''


# Combines the training and label generators
def combine_generator(gen1,gen2):
    while True:
        yield(gen1.next(),gen2.next())


def aug_gen(train,label,bs=32,**gen_arg):
    '''
    Train, label are the training image patches and their labels
    bs - batch size
    gen_arg = - dictionary containing arguments for ImageDataGenerator

    '''

    seed=0 #to do: add this to config file

    #submit your arguments
    aug_patch = ImageDataGenerator(**gen_arg)
    aug_label = ImageDataGenerator(**gen_arg)

    # i get a warning saying that tthe imagedatagenerator calls featurewise
    # centre but i don't?!
    # No harm in including it I guess:
    aug_patch.fit(train, augment=True, seed=seed)
    aug_label.fit(label, augment=True, seed=seed)

    #This is the required form to submit it into the model
    gen_patch  = aug_patch.flow(train,batch_size=bs,seed=seed)
    gen_label  = aug_label.flow(label,batch_size=bs,seed=seed)

    combined = combine_generator(gen_patch, gen_label)
    return combined
