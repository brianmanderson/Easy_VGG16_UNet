'''
Keras_Fine_Tune_VGG_16_Liver.py

This function loads in the original VGG-16 architecture from Keras and appends on the decoding steps are specified by
the given dictionary

Please give reference to https://github.com/brianmanderson/Easy_VGG16_UNet if used
'''
__author__ = 'Brian Mark Anderson'
__email__ = 'bmanderson@mdanderson.org'


from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.layers import Conv2D, Activation, Input, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import get_file
# preprocess_input = imagenet_utils.preprocess_input

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


class Unet(object):
    def define_filters(self, filters=(3,3)):
        self.filters = filters

    def define_activation(self, activation='relu'):
        self.activation = activation

    def define_pool_size(self, pool_size=(2,2)):
        self.pool_size = pool_size

    def conv_block(self, output_size, x, name, strides=1):
        x = Conv2D(output_size, self.filters, activation=None, padding='same',
                   name=name, strides=strides)(x)
        x = Activation(self.activation, name=name+'_activation')(x)
        # x = BatchNormalization()(x)
        return x

    def pool_block(self, x, name):
        return MaxPooling2D(self.pool_size,name=name)(x)

    def make_model(self):
        pass


class VGG_16(Unet):
    def __init__(self, network=None, activation='relu',filter_size=(3,3),out_classes=2):
        self.filter_size = filter_size
        self.define_filters((3,3)) # This changes after the VGG16
        self.define_pool_size((2,2))
        self.define_activation(activation)
        self.out_classes = out_classes
        self.network = network

    def make_network(self):
        filters = 64
        network = {}
        for block in range(5):
            network['Layer_' + str(block)] = {'Encoding': [],'Decoding':[]}
            conv_blocks = 2 if block < 2 else 3
            for conv_block in range(conv_blocks):
                network['Layer_' + str(block)]['Encoding'].append(filters)
            filters *= 2 if filters < 512 else 1
        self.network = network

    def make_model(self):
        image_input_primary = x = Input(shape=(None, None, 3), name='Image_Input')
        layer_vals = {}
        layer_order = []
        layer_index = 0
        for i, layer in enumerate(self.network):
            key = 'Encoding'
            for ii, filters in enumerate(self.network[layer][key]):
                self.desc = layer + '_Encoding_Conv' + str(ii)
                x = self.conv_block(filters,x=x,name='block'+str(i+1)+'_conv'+str(ii+1))
            layer_vals[layer_index] = x
            layer_index += 1
            if layer != 'Layer_4':
                layer_order.append(layer)
                x = self.pool_block(x, name='block'+str(i+1)+'_pool')
        self.define_filters(self.filter_size)
        layer_order.reverse()
        layer_index -= 1
        for i, layer in enumerate(layer_order):
            print(layer)
            layer_index -= 1
            if 'Decoding' not in self.network[layer]:
                continue
            all_filters = self.network[layer]['Decoding']
            x = UpSampling2D(size=self.pool_size, name='Upsampling' + str(i) + '_UNet')(x)
            x = Concatenate(name='concat' + str(i) + '_Unet')([x, layer_vals[layer_index]])
            for i in range(len(all_filters)):
                self.desc = layer + '_Decoding_Conv' + str(i)
                x = self.conv_block(all_filters[i], x, self.desc)

        x = Conv2D(self.out_classes, kernel_size=(1,1), name='Output', activation='softmax')(x)

        model = Model(inputs=[image_input_primary], outputs=[x],name='VGG16_FineTune')
        self.created_model = model

    def load_weights(self):
        weights_path = get_file(
            'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='6d6bbae143d832006294945121d1f1fc')
        self.created_model.load_weights(weights_path, by_name=True)


if __name__ == '__main__':
    pass
