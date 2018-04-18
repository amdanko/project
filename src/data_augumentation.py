# make elastic here
import keras
from keras.preprocessing.image import ImageDataGenerator



'''
This creates data augumentation generator, so my data can be augumented on the fly 
'''
    


def combine_generator(gen1,gen2):
    while True:
        yield(gen1.next(),gen2.next())


def aug_gen(train,label,bs=32,**gen_arg):
    seed=0
    aug_patch = ImageDataGenerator(**gen_arg)
    aug_label = ImageDataGenerator(**gen_arg)

    # i get a warning saying that tthe imagedatagenerator calls featurewise
    # centre but i don't?!
    # no harm in including it i guess
    aug_patch.fit(train, augment=True, seed=seed)
    aug_label.fit(label, augment=True, seed=seed)

    gen_patch  = aug_patch.flow(train,batch_size=bs,seed=seed)
    gen_label  = aug_label.flow(label,batch_size=bs,seed=seed)

    combined = combine_generator(gen_patch, gen_label)
    return combined

