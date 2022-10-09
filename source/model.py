import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Concatenate, UpSampling2D
from tensorflow.keras.models import Model

class Darknet_BN_Leaky(tf.keras.Model):
    def __init__(self, 
                 filter, 
                 kernel, 
                 strides = 1,
                 padding = 'valid'):
        
        super(Darknet_BN_Leaky, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filter,
                                            kernel,
                                            strides = strides,
                                            padding = padding)
                                               
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.lrelu = tf.keras.layers.LeakyReLU()

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.batch_norm(x, training=training)
        x = self.lrelu(x)
        return x

class Res_unit(tf.keras.Model):
    def __init__(self,  
                 filters,
                 strides = 1):
        super(Res_unit, self).__init__()
        self.conv1_1 = Darknet_BN_Leaky(filter = filters,
                                        kernel = 1)

        self.conv1_2 = Darknet_BN_Leaky(filter = filters *2 ,
                                        kernel = 3,
                                        padding = 'same'
                                        )
        

    def call(self, input_tensor):
        
        x = self.conv1_1(input_tensor)
        x = self.conv1_2(x)

        x += input_tensor

        return x 

class ResBlock_N(tf.keras.Model):

    def __init__(self,  
                 filters, 
                 kernel,
                 strides = 1,
                 padding = 'valid'):
        super(ResBlock_N, self).__init__()
        
        self.DBL = tf.keras.layers.Conv2D(filters[0],
                                          kernel,
                                          strides=strides,
                                          padding = padding)

        self.res_body = []

        for ind in range(1, len(filters)):
            self.res_body.append(Res_unit(filters[ind]))

    def call(self, input_tensor):

        x = self.DBL(input_tensor)

        for Res_Unit in self.res_body:
            x = Res_Unit(x)

        return x


def build_model(image_height, image_width, n_classes, n_boxes):

    input_1 = Input(shape = (image_height, image_width, 3), name = 'Input')

    DBL_1 = Darknet_BN_Leaky(32, 3, strides=1, padding= 'same')(input_1)

    res1 = ResBlock_N([64, 32], 3, strides=2, padding='same')(DBL_1)
    res2 = ResBlock_N([128, 64, 64], 3, strides=2, padding='same')(res1)
    res8 = ResBlock_N([256, 128, 128, 128, 128, 128, 128, 128], 3, strides=2, padding='same')(res2)
    res4 = ResBlock_N([512, 256, 256, 256, 256], 3,padding='same')(res8)

    
    DBL_2 = Darknet_BN_Leaky(256, 3, strides=2,padding='same')(res4)
    DBL_3 = Darknet_BN_Leaky(256, 3, padding='same')(DBL_2)
    DBL_4 = Darknet_BN_Leaky(128, 3, padding='same')(DBL_3)
    DBL_5 = Darknet_BN_Leaky(64, 3, padding='same')(DBL_4)
    DBL_6 = Darknet_BN_Leaky(32, 3,strides=2, padding='same')(DBL_5)

    DBL_7 = Darknet_BN_Leaky(n_boxes*(4+n_classes), 3, padding='same')(DBL_6)
#    boxes_1 = Reshape((-1, (5+n_classes)), name='boxes1_reshape')(DBL_7)


#    DBL_8 = Darknet_BN_Leaky(256, 3, padding='same')(DBL_6)
#    up = UpSampling2D((2,2), name='up1')(DBL_8)
#    concat_1 = Concatenate()([up, res8])
#    DBL_9 = Darknet_BN_Leaky((5+n_classes), 3, padding='same')(concat_1)
#    boxes_2 =  Reshape((-1, (5+n_classes)), name='boxes2_reshape')(DBL_9)
    
#    output = Concatenate(axis=1)([boxes_1, boxes_2])
    model = Model(inputs = input_1, outputs = DBL_7)

    return model


