# -*- coding=utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import UpSampling2D, Concatenate, Conv2D, BatchNormalization, Activation, Conv2DTranspose, MaxPooling2D
DEFAULT_SKIP_CONNECTIONS = {
    'VGG16':            ('block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'),
    'vgg19':            ('block5_conv4', 'block4_conv4', 'block3_conv4', 'block2_conv2', 'block1_conv2'),
    'resnet18':         ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'), # check 'bn_data'
    'resnet34':         ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnet50':         ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnet101':        ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnet152':        ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnext50':        ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnext101':       ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'inceptionv3':          (228, 86, 16, 9),
    'inceptionresnetv2':    (594, 260, 16, 9),
    'densenet121':          (311, 139, 51, 4),
    'densenet169':          (367, 139, 51, 4),
    'densenet201':          (479, 139, 51, 4),
}


def get_layer_number(model, layer_name):
    """
    Help find layer in Keras model by name
    Args:
        model: Keras `Model`
        layer_name: str, name of layer

    Returns:
        index of layer

    Raises:
        ValueError: if model does not contains layer with such name
    """
    for i, l in enumerate(model.layers):
        if l.name == layer_name:
            return i
    raise ValueError('No layer with name {} in  model {}.'.format(layer_name, model.name))


def to_tuple(x):
    if isinstance(x, tuple):
        if len(x) == 2:
            return x
    elif np.isscalar(x):
        return (x, x)
    raise ValueError('Value should be tuple of length 2 or int value, got "{}"'.format(x))


def handle_block_names(stage):
    conv_name = 'decoder_stage{}_conv'.format(stage)
    bn_name = 'decoder_stage{}_bn'.format(stage)
    relu_name = 'decoder_stage{}_relu'.format(stage)
    up_name = 'decoder_stage{}_upsample'.format(stage)
    return conv_name, bn_name, relu_name, up_name


def ConvRelu(filters, kernel_size, use_batchnorm=False, conv_name='conv', bn_name='bn', relu_name='relu'):
    def layer(x):
        x = Conv2D(filters, kernel_size, padding="same", name=conv_name, use_bias=not(use_batchnorm))(x)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name)(x)
        x = Activation('relu', name=relu_name)(x)
        return x
    return layer


def Upsample2D_block(filters, stage, kernel_size=(3,3), upsample_rate=(2,2),
                     use_batchnorm=False, skip=None):

    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)
        print(input_tensor)
        x = UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)

        if skip is not None:
            x = Concatenate()([x, skip])

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '1', bn_name=bn_name + '1', relu_name=relu_name + '1')(x)

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)

        return x
    return layer


def Transpose2D_block(filters, stage, kernel_size=(3,3), upsample_rate=(2,2),
                      transpose_kernel_size=(4,4), use_batchnorm=False, skip=None):

    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)

        x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,
                            padding='same', name=up_name, use_bias=not(use_batchnorm))(input_tensor)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name+'1')(x)
        x = Activation('relu', name=relu_name+'1')(x)

        if skip is not None:
            x = Concatenate()([x, skip])

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)

        return x
    return layer


class EncoderSimple(tf.keras.Model):
    def __init__(self, model_name):
        super(EncoderSimple, self).__init__(name='Encoder')
        self.conv2d1 = Conv2D(8, (3, 3), activation='relu', padding='same')
        self.maxpooling2d1 = MaxPooling2D((2, 2), padding='same')
        self.conv2d2 = Conv2D(8, (3, 3), activation='relu', padding='same')
        self.maxpooling2d2 = MaxPooling2D((2, 2), padding='same')
        self.conv2d3 = Conv2D(16, (3, 3), activation='relu', padding='same')
        self.maxpooling2d3 = MaxPooling2D((2, 2), padding='same')

    def call(self, inputs):
        x = self.conv2d1(inputs)
        x = self.maxpooling2d1(x)
        x = self.conv2d2(x)
        x = self.maxpooling2d2(x)
        x = self.conv2d3(x)
        return self.maxpooling2d3(x)

    def compute_output_shape(self, input_shape):
        print('compute_output_shape')


class Encoder(tf.keras.Model):
    def __init__(self, model_name):
        super(Encoder, self).__init__(name='Encoder')
        self.model_name = model_name
        if model_name == 'VGG16':
            self.base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')


    def call(self, inputs):
        final_feature_map = self.base_model(inputs)
        skip_connection_names = DEFAULT_SKIP_CONNECTIONS[self.model_name]
        skip_connection_inputs = []
        for skip_connection_name in skip_connection_names:
            skip_connection_inputs.append(get_layer_number(self.base_model, skip_connection_name))
        return final_feature_map, skip_connection_inputs

    def compute_output_shape(self, input_shape):
        print('compute_output_shape')


class DecoderSimple(tf.keras.Model):
    def __init__(self, n_upsample_blocks=5, block_type='upsampling', decoder_filters=(256, 128, 64, 32, 16),
                 upsampling_rate=(2, 2, 2, 2, 2), use_batchnorm=False):
        '''
        decoder 模型
        :param base_mode: Encoder的基础模型
        :param model_stride: Encoder的步长
        '''
        super(DecoderSimple, self).__init__(name='Decoder')
        self.conv2d1 = Conv2D(8, (3, 3), activation='relu', padding='same')
        self.upsampling2d1 = UpSampling2D((2, 2))
        self.conv2d2 = Conv2D(8, (3, 3), activation='relu', padding='same')
        self.upsampling2d2 = UpSampling2D((2, 2))
        self.conv2d3 = Conv2D(16, (3, 3), activation='relu')
        self.upsampling2d3 = UpSampling2D((2, 2))
        self.conv2d4 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')

    def call(self, inputs):
        x = self.conv2d1(inputs)
        x = self.upsampling2d1(x)
        x = self.conv2d2(x)
        x = self.upsampling2d2(x)
        x = self.conv2d3(x)
        x = self.upsampling2d3(x)
        decoded = self.conv2d4(x)
        return decoded


class Decoder(tf.keras.Model):
    def __init__(self, n_upsample_blocks=5, block_type='upsampling', decoder_filters=(256, 128, 64, 32, 16),
                 upsampling_rate=(2, 2, 2, 2, 2), use_batchnorm=False):
        '''
        decoder 模型
        :param base_mode: Encoder的基础模型
        :param model_stride: Encoder的步长
        '''
        super(Decoder, self).__init__(name='Decoder')
        if block_type == 'upsampling':
            up_block = Upsample2D_block
        elif block_type == 'transpose':
            up_block = Transpose2D_block
        else:
            print('the type of block_type is not supported!')
            assert False
        self.n_upsampling_block = n_upsample_blocks
        self.up_blocks = []
        for i in range(n_upsample_blocks):
            self.up_blocks.append(
                up_block(decoder_filters[i], i, upsample_rate=to_tuple(upsampling_rate[i]), use_batchnorm=use_batchnorm,
                         skip=None))

    def call(self, inputs):
        x = inputs[0]
        for i in range(self.n_upsampling_block):
            x = self.up_blocks[i](x)
        return x



class AutoEncoder(tf.keras.Model):
    def __init__(self, base_model, output_dim=3, logits_activation='sigmoid'):
        super(AutoEncoder, self).__init__(name='AutoEncoder')
        self.encoder_model = Encoder(base_model)
        self.decoder_model = Decoder()
        self.output_dim = output_dim
        self.logits_activation = logits_activation

    def call(self, inputs):
        inputs = tf.image.resize_images(inputs, [128, 128])
        final_feature_map, skip_connection_inputs = self.encoder_model(inputs)
        decoder_output = self.decoder_model([final_feature_map, skip_connection_inputs])
        return self.logits_layer(decoder_output, self.logits_activation)

    def logits_layer(self, x, activation='sigmoid'):
        x = Conv2D(self.output_dim, kernel_size=(3, 3), padding='SAME', name='logits')(x)
        x = tf.image.resize_images(x, [28, 28])
        if activation is not None:
            x = Activation(activation, name=activation)(x)

        # x = layers.AlphaDropout(rate=0.5)(x)
        return x


class AutoEncoderSimple(tf.keras.Model):
    def __init__(self, base_model, output_dim=3, logits_activation=None):
        super(AutoEncoderSimple, self).__init__(name='AutoEncoder')
        self.encoder_model = EncoderSimple(base_model)
        self.decoder_model = DecoderSimple()
        self.output_dim = output_dim
        self.logits_activation = logits_activation

    def call(self, inputs):
        final_feature_map = self.encoder_model(inputs)
        decoder_output = self.decoder_model(final_feature_map)
        return decoder_output


if __name__ == '__main__':
    config = tf.ConfigProto()
    from keras.backend.tensorflow_backend import set_session
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(tf.Session(config=config))

    from tensorflow.keras.datasets import mnist
    from tensorflow.keras import backend as K
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_rows, img_cols = 28, 28
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    x_test = np.concatenate([x_test, x_test, x_test], axis=-1)
    x_train = np.concatenate([x_train, x_train, x_train], axis=-1)
    # data = np.concatenate([x_train, x_train, x_train], axis=-1)

    autoencoder_model = AutoEncoder('VGG16', logits_activation='sigmoid')
    autoencoder_model.compile(tf.train.AdadeltaOptimizer(learning_rate=0.1), loss='mse', metrics=['mse'])
    # autoencoder_model.compile('adam', loss='mse', metrics=['mse'])
    autoencoder_model.fit(x_train, x_train, epochs=2, batch_size=128, validation_data=(x_test, x_test), shuffle=True)
    autoencoder_model.save_weights('./weights/my_model')
    autoencoder_model.load_weights('./weights/my_model')
    print('score is ', autoencoder_model.evaluate(x_test, x_test, batch_size=128, verbose=1))
    predict_result = autoencoder_model.predict(x_test, batch_size=128, verbose=1)
    print(np.max(predict_result), np.min(predict_result))
    print(np.max(x_test), np.min(x_test))
    print('the mse is ', np.mean(np.sum((x_test - predict_result) ** 2, axis=3)))
    import matplotlib.pyplot as plt
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(1, n+1):
        # display original
        ax = plt.subplot(2, n, i)
        plt.imshow(x_test[i][:, :, 1].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(predict_result[i][:, :, 1].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig('./weights/mnist.png')
    # plt.show()
