"# Easy_VGG16_UNet" 

This works by loading in the pre-trained weights on the encoding side of the VGG-16 architecture

    from Keras_Fine_Tune_VGG_16_Liver import VGG_16
    network = {'Layer_0': {'Encoding': [64, 64], 'Decoding': [64, 32]},
               'Layer_1': {'Encoding': [128, 128], 'Decoding': [128]},
               'Layer_2': {'Encoding': [256, 256, 256], 'Decoding': [256]},
               'Layer_3': {'Encoding': [512, 512, 512], 'Decoding': [512]},
               'Layer_4': {'Encoding': [512, 512, 512]}}
    VGG_model = VGG_16(network=network, activation='relu',filter_size=(3,3))
    VGG_model.make_model()
    VGG_model.load_weights()
    new_model = VGG_model.created_model
    VGG_model.created_model.save(my_path)
The 'Decoding' variables are available to be changed easy in this way
