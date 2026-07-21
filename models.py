"""Shared building blocks and 5 diverse CNN architectures."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPool2D, GlobalAveragePooling2D, Dropout, SpatialDropout2D,
    Dense, BatchNormalization, Activation, Add, Input, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import config


def conv_bn_relu(x, filters, kernel_size, l2=0.001, sdrop=0.1):
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(l2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if sdrop > 0:
        x = SpatialDropout2D(sdrop)(x)
    return x


def dense_bn_relu(x, units, drop, l2=0.001):
    x = Dense(units, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(drop)(x)
    return x


def make_output_head(x, n_classes=config.N_CLASSES):
    x = Dense(n_classes, dtype='float32')(x)
    return Activation('softmax', dtype='float32')(x)


def compile_model(model, lr=0.001, wd=1e-4, label_smooth=0.1):
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smooth)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def get_callbacks(path, patience_es=40, patience_lr=15):
    return [
        EarlyStopping(monitor='val_accuracy', patience=patience_es, mode='max',
                      restore_best_weights=True, verbose=0),
        ModelCheckpoint(path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience_lr,
                          min_lr=1e-6, verbose=1),
    ]


def build_cnn_a(input_shape=config.INPUT_SHAPE, n_classes=config.N_CLASSES):
    """Shallow: 2 conv blocks."""
    inp = Input(shape=input_shape, name='A_input')
    x = conv_bn_relu(inp, 64, (3, 2), sdrop=0.1)
    x = conv_bn_relu(x, 64, (3, 2), sdrop=0.0)
    x = MaxPool2D((2, 1))(x)
    x = conv_bn_relu(x, 128, (3, 2), sdrop=0.1)
    x = conv_bn_relu(x, 128, (3, 2), sdrop=0.0)
    x = MaxPool2D((2, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = dense_bn_relu(x, 128, drop=0.4)
    x = dense_bn_relu(x, 64, drop=0.3)
    out = make_output_head(x, n_classes)
    return compile_model(Model(inp, out, name='CNN_A'))


def build_cnn_b(input_shape=config.INPUT_SHAPE, n_classes=config.N_CLASSES):
    """Deep: 4 conv blocks, progressive filters 64->128->256->256."""
    inp = Input(shape=input_shape, name='B_input')
    x = conv_bn_relu(inp, 64, (3, 2), sdrop=0.1)
    x = MaxPool2D((2, 1))(x)
    x = conv_bn_relu(x, 128, (3, 2), sdrop=0.1)
    x = MaxPool2D((2, 1))(x)
    x = conv_bn_relu(x, 256, (3, 2), sdrop=0.1)
    x = conv_bn_relu(x, 256, (3, 2), sdrop=0.0)
    x = MaxPool2D((2, 1))(x)
    x = conv_bn_relu(x, 256, (3, 2), sdrop=0.0)
    x = GlobalAveragePooling2D()(x)
    x = dense_bn_relu(x, 256, drop=0.4)
    x = dense_bn_relu(x, 128, drop=0.3)
    x = dense_bn_relu(x, 64, drop=0.3)
    out = make_output_head(x, n_classes)
    return compile_model(Model(inp, out, name='CNN_B'), lr=0.0008)


def build_cnn_c(input_shape=config.INPUT_SHAPE, n_classes=config.N_CLASSES):
    """Wide: larger (5,2) kernels for a bigger receptive field."""
    inp = Input(shape=input_shape, name='C_input')
    x = conv_bn_relu(inp, 64, (5, 2), l2=0.0005, sdrop=0.1)
    x = conv_bn_relu(x, 64, (3, 2), l2=0.0005, sdrop=0.0)
    x = MaxPool2D((2, 1))(x)
    x = conv_bn_relu(x, 128, (5, 2), l2=0.0005, sdrop=0.1)
    x = conv_bn_relu(x, 128, (3, 2), l2=0.0005, sdrop=0.0)
    x = MaxPool2D((2, 1))(x)
    x = conv_bn_relu(x, 256, (3, 2), l2=0.0005, sdrop=0.1)
    x = MaxPool2D((2, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = dense_bn_relu(x, 256, drop=0.35, l2=0.0005)
    x = dense_bn_relu(x, 128, drop=0.3, l2=0.0005)
    out = make_output_head(x, n_classes)
    return compile_model(Model(inp, out, name='CNN_C'), lr=0.001, label_smooth=0.05)


def residual_block(x, filters, kernel_size=(3, 2), l2=0.001):
    """ResNet-style block with 1x1 projection shortcut."""
    shortcut = x
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(l2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(l2))(x)
    x = BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding='same',
                          kernel_regularizer=regularizers.l2(l2))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def build_cnn_d(input_shape=config.INPUT_SHAPE, n_classes=config.N_CLASSES):
    """Residual skip connections."""
    inp = Input(shape=input_shape, name='D_input')
    x = conv_bn_relu(inp, 64, (3, 2), sdrop=0.0)
    x = residual_block(x, 64)
    x = MaxPool2D((2, 1))(x)
    x = SpatialDropout2D(0.1)(x)
    x = residual_block(x, 128)
    x = MaxPool2D((2, 1))(x)
    x = SpatialDropout2D(0.1)(x)
    x = residual_block(x, 256)
    x = SpatialDropout2D(0.1)(x)
    x = GlobalAveragePooling2D()(x)
    x = dense_bn_relu(x, 256, drop=0.4)
    x = dense_bn_relu(x, 128, drop=0.3)
    out = make_output_head(x, n_classes)
    return compile_model(Model(inp, out, name='CNN_D'), lr=0.0008, wd=5e-5)


def inception_block(x, filters):
    """Three parallel convolutions: 1x1, 3x2, 5x2, concatenated."""
    f = filters // 3
    b1 = Conv2D(f, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    b2 = Conv2D(f, (3, 2), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    b3 = Conv2D(f, (5, 2), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    out = Concatenate()([b1, b2, b3])
    out = BatchNormalization()(out)
    return out


def build_cnn_e(input_shape=config.INPUT_SHAPE, n_classes=config.N_CLASSES):
    """Multi-scale inception-style."""
    inp = Input(shape=input_shape, name='E_input')
    x = conv_bn_relu(inp, 32, (3, 2), sdrop=0.0)
    x = inception_block(x, 96)
    x = SpatialDropout2D(0.1)(x)
    x = MaxPool2D((2, 1))(x)
    x = inception_block(x, 192)
    x = SpatialDropout2D(0.1)(x)
    x = MaxPool2D((2, 1))(x)
    x = inception_block(x, 192)
    x = SpatialDropout2D(0.1)(x)
    x = GlobalAveragePooling2D()(x)
    x = dense_bn_relu(x, 256, drop=0.4)
    x = dense_bn_relu(x, 128, drop=0.3)
    out = make_output_head(x, n_classes)
    return compile_model(Model(inp, out, name='CNN_E'), lr=0.001, label_smooth=0.08)


BUILDERS = {
    'CNN_A': build_cnn_a,
    'CNN_B': build_cnn_b,
    'CNN_C': build_cnn_c,
    'CNN_D': build_cnn_d,
    'CNN_E': build_cnn_e,
}
