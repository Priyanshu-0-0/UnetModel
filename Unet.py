
import tensorflow as tf

tf.keras.mixed_precision.set_global_policy('mixed_float16')
def unet_model(input,n_classes):

    # Input Shape


    input_shape=input[0].shape

    # Input Layer

    input_layer=tf.keras.Input(shape=(input_shape))

    # Encoder

    # Conv BLock 1 3x3 Relu

    conv1=tf.keras.layers.Conv2D(64,(3,3), activation="relu", kernel_initializer="he_normal", padding="same")(input_layer)
    conv2=tf.keras.layers.Conv2D(64,(3,3), activation="relu", kernel_initializer="he_normal", padding="same")(conv1)

    # Maxpooling LAyer 1

    mx1=tf.keras.layers.MaxPool2D((2,2))(conv2)

    # Conv BLock 2 3x3 Relu

    conv3=tf.keras.layers.Conv2D(128,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(mx1)
    conv4=tf.keras.layers.Conv2D(128,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(conv3)

    # Maxpooling LAyer 1

    mx2=tf.keras.layers.MaxPool2D((2,2))(conv4)

    # Conv BLock 3 3x3 Relu

    conv5=tf.keras.layers.Conv2D(256,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(mx2)
    conv6=tf.keras.layers.Conv2D(256,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(conv5)

    # Maxpooling LAyer 1

    mx3=tf.keras.layers.MaxPool2D((2,2))(conv6)

    # Conv BLock 4 3x3 Relu

    conv7=tf.keras.layers.Conv2D(512,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(mx3)
    conv8=tf.keras.layers.Conv2D(512,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(conv7)

    # Maxpooling LAyer 1

    mx4=tf.keras.layers.MaxPool2D((2,2))(conv8)

    # Conv Block 5 3x3 Relu

    conv9=tf.keras.layers.Conv2D(1024,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(mx4)
    conv10=tf.keras.layers.Conv2D(1024,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(conv9)

    # Decoder

    #  Conv Block 4  + ConvTranspose 1  2x2

    conv11=tf.keras.layers.Conv2DTranspose(1024,(2,2),strides=(2,2),padding="same")(conv10)
    conv11=tf.keras.layers.concatenate([conv8,conv11])


    # Conv BLock 6 3x3 Relu

    conv12=tf.keras.layers.Conv2D(512,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(conv11)
    conv13=tf.keras.layers.Conv2D(512,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(conv12)

    # Conv Block 3 + ConvTranspose 2  2x2

    conv14=tf.keras.layers.Conv2DTranspose(512,(2,2),strides=(2,2),padding="same")(conv13)
    conv14=tf.keras.layers.concatenate([conv6,conv14])

    # Conv BLock 7 3x3 Relu

    conv15=tf.keras.layers.Conv2D(256,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(conv14)
    conv16=tf.keras.layers.Conv2D(256,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(conv15)

    #  Conv Block 2 + ConvTranspose 3  2x2

    conv17=tf.keras.layers.Conv2DTranspose(256,(2,2),strides=(2,2),padding="same")(conv16)
    conv17=tf.keras.layers.concatenate([conv4,conv17])

    # Conv BLock 8 3x3 Relu

    conv18=tf.keras.layers.Conv2D(128,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(conv17)
    conv19=tf.keras.layers.Conv2D(128,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(conv18)

    #  Conv Block 1  + ConvTranspose 4  2x2

    conv20=tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding="same")(conv19)
    conv20=tf.keras.layers.concatenate([conv2,conv20])

    # Conv BLock 9  ( 2 3x3 Relu Conv)

    conv21=tf.keras.layers.Conv2D(64,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(conv20)
    conv22=tf.keras.layers.Conv2D(64,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(conv21)

    # Output Layer

    output_layer = tf.keras.layers.Conv2D(n_classes,(1,1),activation="softmax")(conv22)

    # Model

    model=tf.keras.Model(inputs=[input_layer],outputs=[output_layer])

    #return model

    return model
