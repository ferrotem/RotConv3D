#%%
import tensorflow as tf 
from tensorflow.keras.layers import Lambda, BatchNormalization, Conv3D, Dropout, Add, Input
from tensorflow.keras.layers import GlobalAveragePooling3D ,Dense, Flatten,  MaxPooling3D, Activation, Dropout
import config as cfg

class Linear(tf.keras.layers.Layer):
    def __init__(self, units=3, input_dim=1213056):#139968
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer( mean=0.0, stddev=0.05, seed=1313)
        self.w = tf.Variable(
            initial_value=w_init(shape=(units, input_dim), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,input_dim), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        return tf.math.multiply(inputs, self.w)+ self.b #tf.matmul(inputs, self.w) 


class Model:

    def __init__(self):
        self.network =  self.Rot3D()
        self.resnet = self.resnet_3D_model(cfg.INPUT_SHAPE_Z, "just_res")


    def res_net_block(self, input_data, filters, conv_size):
        if cfg.DILATION:
            x = Conv3D(filters, conv_size, activation='relu', padding='same', dilation_rate=[2,2,2])(input_data)
        x = Conv3D(filters, conv_size, activation='relu', padding='same')(input_data)
        x = BatchNormalization()(x)
        if cfg.DILATION:
             x = Conv3D(filters, conv_size, activation=None, padding='same', dilation_rate=[2,2,2])(x)
        x = Conv3D(filters, conv_size, activation=None, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x, input_data])
        x = Dropout(rate=0.3)(x)
        x = Activation('relu')(x)
        # print("x shape : ", x.shape)    
        return x    

    def resnet_3D_model(self,input_shape, net_name,  num_res_net_blocks = 5):
        inputs = Input(input_shape)
        x = Conv3D(32, 3, activation='relu')(inputs) # 1st dimention of filter is for order of frames
        # print ("x shape : ", x.shape)
        x = MaxPooling3D((2,2,2))(x)
        x = Conv3D(32, 3, activation='relu')(x) # 1st dimention of filter is for order of frames
        x = MaxPooling3D((2,2,2))(x)
        # print ("x shape : ", x.shape)
        # # 

        for _ in range(cfg.NUMBER_OF_RES_BLOCKS):
            x = self.res_net_block(x, 32, 3)

        # print ("x shape : ", x.shape)
        # outputs = MaxPooling3D((2,2,2))(x)
        # outputs = Flatten()(x)
        #outputs = Dense(80,activation='relu',use_bias=False)(x)  
        return tf.keras.Model(inputs, x, name= net_name)


    def Rot3D(self):

        Z = Input(cfg.INPUT_SHAPE_Z, name = "Original_input")
        Y = tf.transpose(Z, perm=[0,2,1,3,4], name= 'Y')
        X = tf.transpose(Z, perm=[0,2,3,1,4], name = 'X')

        resnet_Z = self.resnet_3D_model(cfg.INPUT_SHAPE_Z, "net_Z")
        resnet_Y = self.resnet_3D_model(cfg.INPUT_SHAPE_Y, "net_Y")
        resnet_X = self.resnet_3D_model(cfg.INPUT_SHAPE_X, "net_X")

        Z_out = resnet_Z(Z)
        Y_out = resnet_Y(Y)
        X_out = resnet_X(X)

        reverse_Y = tf.transpose(Y_out, perm=[0,2,1,3,4], name= 'Y_reverse')    
        reverse_X = tf.transpose(X_out, perm=[0,3,1,2,4], name = 'X_reverse')



        z = Flatten()(Z_out)
        y = Flatten()(reverse_Y)
        x = Flatten()(reverse_X)

        out = tf.stack([z,y,x], axis=1) 
        print("out_shape, ", out.shape)
        
        out = Linear()(out)
        
        # z = Linear()(z)
        # y = Linear()(y)
        # x = Linear()(x)     

        # out = tf.concat([z,x,y], axis = 1, name="Concat")
        out = tf.nn.elu(out, name="ELU")
        
        
        # out = tf.stack([z,y,x], axis=2) 
        # out =  Lambda(lambda x: tf.reduce_max(x, axis=2))(out)
        # out = tf.concat([z,x,y], axis = 1)
        out = Dense(80,activation='sigmoid',use_bias=True)(out) 
        return tf.keras.Model(inputs=[Z],outputs=[out, z,y,x])
#%%
# resnet_Z = Res3D(cfg.INPUT_SHAPE_Z).network
# resnet_Y = Model(cfg.INPUT_SHAPE_Y).R3D
# resnet_X = Model(cfg.INPUT_SHAPE_Z).R3D

# def call_network(input_shape, inputs):
#     net = Res3D(input_shape).network
#     return Flatten()(net(inputs))


# def Rot3D():

# Z = Input(cfg.INPUT_SHAPE_Z, name = "Original_input")
# X = tf.transpose(Z, perm=[0,2,3,1,4], name = 'X')
# Y = tf.transpose(Z, perm=[0,2,1,3,4], name= 'Y')

# resnet_Z = Res3D(cfg.INPUT_SHAPE_Z, "net_Z").network
# resnet_Y = Res3D(cfg.INPUT_SHAPE_Y, "net_X").network
# resnet_X = Res3D(cfg.INPUT_SHAPE_X, "net_Y").network

# z = Flatten()(resnet_Z(Z))
# x = Flatten()(resnet_X(X))
# y = Flatten()(resnet_Y(Y))

# out = tf.concat([z,x,y], axis = 1)
# out = Dense(80,activation='relu',use_bias=False)(out) 
# return tf.keras.Model(inputs=[Z],outputs=[out])                                        




