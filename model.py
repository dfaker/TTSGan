
from keras.applications.vgg19 import VGG19,preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.layers import Conv2D,Input,Add,BatchNormalization,Concatenate,UpSampling2D,Flatten,Reshape,Dense,Add,Dropout,ZeroPadding2D
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU

from keras.initializers import RandomNormal

letters = ' DHPTCIYKBLXRWAQVGZNFSMUEOJ#'

conv_init = RandomNormal(0, 0.02)

def getUpscaler(layers=5):
  x_in = Input(shape=(200,200,1))

  #xu = Dense(4*4*1024)(x_in)
  #xu = Reshape((4,4,1024))(xu)

  xu = Flatten()(x_in)
  xu = Dense(4*4*2*64*2,name='dense0')(xu)
  xu = Reshape((4,4*2,64*2),name='rehape0')(xu)

  x = BatchNormalization(momentum=0.8,name='bn0')(xu)

  depth = 1024*2
  if layers>=1:
    x = UpSampling2D(name='us1')(x)
    depth=depth//2
    x = Conv2D(depth, kernel_size=3, padding="same",name='conv1',activation='relu')(x)
    x = BatchNormalization(momentum=0.8,name='bn1')(x)

  
  if layers>=2:
    x = UpSampling2D(name='us2')(x) 
    depth=depth//2
    x = Conv2D(depth, kernel_size=3, padding="same",name='conv2',activation='relu')(x)
    x = BatchNormalization(momentum=0.8,name='bn2')(x)

  
  if layers>=3:
    x = UpSampling2D(name='us3')(x)
    depth=depth//2
    x = Conv2D(depth, kernel_size=3, padding="same",name='conv3',activation='relu')(x)
    x = BatchNormalization(momentum=0.8,name='bn3')(x)

  if layers>=4:
    x = UpSampling2D(name='us4')(x)
    depth=depth//2
    x = Conv2D(depth, kernel_size=3, padding="same",name='conv4',activation='relu')(x)
    x = BatchNormalization(momentum=0.8,name='bn4')(x)

  if layers>=5:
    x = UpSampling2D(name='us5')(x)
    depth=depth//2
    x = Conv2D(depth, kernel_size=3, padding="same",name='conv5',activation='relu')(x)
    x = BatchNormalization(momentum=0.8,name='bn5')(x)

  x = UpSampling2D(name='us6')(x)
  depth=depth//2
  x = Conv2D(depth, kernel_size=3, padding="same",name='conv6',activation='relu')(x)
  x = BatchNormalization(momentum=0.8,name='bn6')(x)
  depth=depth//2
  x = Conv2D(depth, kernel_size=3, padding="same",name='conv7',activation='relu')(x)
  x = BatchNormalization(momentum=0.8)(x)

  x = UpSampling2D(name='us7')(x)  # 96x128 -> 192x256
  depth=depth//2
  x = Conv2D(depth, kernel_size=1, strides=(2,2), padding="same",name='conv8',activation='relu')(x)

  x = Conv2D( 1, kernel_size=5, padding='same', activation='sigmoid',name='conv9' )(x)

  upscaler = Model(x_in,x)
  return upscaler

def getDescrim(inputShape):
  print(inputShape)
  cond_in = Input(shape=(200,200,1))

  cu = Flatten()(cond_in)
  cu = Dense(8*16*1,name='dense0')(cu)
  cu = Reshape((8, 16, 1),name='rehape0')(cu)
  cu = UpSampling2D(size=32)(cu)


  a_in = Input(shape=(inputShape, inputShape*2, 1))

  x = Concatenate()([a_in,cu])

  x = Conv2D(32, kernel_size=3, strides=2, padding="same")(x)  # 192x256 -> 96x128
  x = LeakyReLU(alpha=0.2)(x)
  x = Dropout(0.25)(x)

  x = Conv2D(64, kernel_size=3, strides=2, padding="same")(x)  # 96x128 -> 48x64
  x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
  x = LeakyReLU(alpha=0.2)(x)
  x = Dropout(0.25)(x)
  x = BatchNormalization(momentum=0.8)(x)

  x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)  # 48x64 -> 24x32
  x = LeakyReLU(alpha=0.2)(x)
  x = Dropout(0.25)(x)
  x = BatchNormalization(momentum=0.8)(x)

  x = Conv2D(256, kernel_size=3, strides=1, padding="same")(x)  # 24x32 -> 12x16
  x = LeakyReLU(alpha=0.2)(x)
  x = Dropout(0.25)(x)

  x = Conv2D(512, kernel_size=3, strides=1, padding="same")(x)  # 12x16 -> 6x8
  x = LeakyReLU(alpha=0.2)(x)
  x = Flatten()(x)
  
  validity = Dense(1,activation='sigmoid')(x) 

  descrim = Model([cond_in,a_in],validity)
  return descrim

from tensorflow.keras.optimizers import Adam
g_optimizer = Adam(0.0002, 0.5)
d_optimizer = Adam(0.0002, 0.5)


def getJoinedModel(upscaleLayers=5):

  upscaler = getUpscaler(upscaleLayers)
  upscaler.summary()
  upscaler.compile(g_optimizer,'mse')

  descrim = getDescrim( upscaler.output.shape[1] )
  descrim.compile(d_optimizer,'binary_crossentropy')
  descrim.summary()

  for layer in descrim.layers:
    layer.trainable=False
  descrim.trainable=False


  combined = Model([upscaler.input], descrim([upscaler.input,upscaler.output])  )
  combined.compile(g_optimizer,'binary_crossentropy')

  return upscaler,descrim,combined

if __name__ == '__main__':
  getJoinedModel()