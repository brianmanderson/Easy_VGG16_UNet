'''
Keras_Fine_Tune_VGG_16_Liver.py

This function loads in the original VGG-16 architecture from Keras (Do not change the encoding side) and appends on
the decoding side to the original resolution

Please give reference to https://github.com/brianmanderson/Easy_VGG16_UNet if used
'''
__author__ = 'Brian Mark Anderson'
__email__ = 'bmanderson@mdanderson.org'

from Keras_Fine_Tune_VGG_16_Liver import VGG_16

my_path = r'Y:\CNN\VGG_16\VGG16_UNet.hdf5'
network = {'Layer_0': {'Encoding': [64, 64], 'Decoding': [64, 32]},
           'Layer_1': {'Encoding': [128, 128], 'Decoding': [128]},
           'Layer_2': {'Encoding': [256, 256, 256], 'Decoding': [256]},
           'Layer_3': {'Encoding': [512, 512, 512], 'Decoding': [512]},
           'Layer_4': {'Encoding': [512, 512, 512]}}
'''
The encoding part of the network should not be changed, as it will not load in the VGG_16 pre-trained weights otherwise
'''
VGG_model = VGG_16(network=network, activation='relu',filter_size=(3,3))
VGG_model.make_model()
VGG_model.load_weights()
new_model = VGG_model.created_model
VGG_model.created_model.save(my_path)