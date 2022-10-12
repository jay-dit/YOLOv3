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


    def get_config(self):
        config = super(Darknet_BN_Leaky, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

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

    def get_config(self):
        config = super(Res_unit, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

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
    def get_config(self):
        config = super(ResBlock_N, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
class Yolo(tf.losses.Loss):
    """Implements Yolo loss"""

    def __init__(self, lambda_1, lambda_2):
        super(Yolo, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def call(self, y_true, y_pred):

        mask = y_true[:,:,4] > 0.5
        
        class_loss =  tf.reduce_sum((y_true[:,:,4] - y_pred[:,:,4])**4)
        
        difference_x = self.lambda_1*tf.reduce_sum((y_true[:,:,0][mask] - y_pred[:,:,0][mask])**2)
        

        difference_y = self.lambda_1*tf.reduce_sum((y_true[:,:,1][mask] - y_pred[:,:,1][mask])**2)
        
        difference_w = self.lambda_2*tf.reduce_sum((y_true[:,:,2][mask] - y_pred[:,:,2][mask])**2)
        
        difference_h = self.lambda_2*tf.reduce_sum((y_true[:,:,3][mask] - y_pred[:,:,3][mask])**2)

        
        return class_loss + difference_x + difference_y + difference_h + difference_w

    def get_config(self):
        config = super(Yolo, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
        

def build_model(image_height, image_width, n_classes, n_boxes):

    input_1 = Input(shape = (image_height, image_width, 3), name = 'Input')

    DBL_1 = Darknet_BN_Leaky(32, 3, strides=1, padding= 'same')(input_1)

    res1 = ResBlock_N([64, 32], 3, strides=2, padding='same')(DBL_1)
    res2 = ResBlock_N([128, 64, 64], 3, strides=2, padding='same')(res1)
    res8 = ResBlock_N([256, 128, 128, 128, 128, 128, 128, 128], 3, strides=1, padding='same')(res2)
    res4 = ResBlock_N([512, 256, 256, 256, 256], 3,padding='same')(res8)

    
    DBL_2 = Darknet_BN_Leaky(256, 3,padding='same')(res4)
    DBL_3 = Darknet_BN_Leaky(256, 3, padding='same')(DBL_2)
    DBL_4 = Darknet_BN_Leaky(128, 3, strides=2, padding='same')(DBL_3)
    DBL_5 = Darknet_BN_Leaky(64, 3, padding='same')(DBL_4)
    DBL_6 = Darknet_BN_Leaky(32, 3, padding='same')(DBL_5)

    output = Darknet_BN_Leaky(n_boxes*(4+n_classes), 3, padding='same')(DBL_6)
    model = Model(inputs = input_1, outputs = output)

    return model


