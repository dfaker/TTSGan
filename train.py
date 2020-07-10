
import model
import random
import glob
import numpy as np
import cv2

generator,discriminator,combined = model.getJoinedModel(upscaleLayers=5)
imageCache = {}

try:
  discriminator.load_weights('discriminator.bin')
  generator.load_weights('generator.bin')
except Exception as e:
  print(e)

def genPairs():

  sources= [x for x in list(glob.glob('sourceSamples\\*.png')) if '\\Zaud' not in x]
  
  while 1:
    random.shuffle(sources)
    for source in sources:

      name = source.split('\\')[-1]
      name = name.replace('.png','').strip().upper().replace('[MUSIC]','#')
      inds = [model.letters.index(x) if x in model.letters else 0 for x in name]


      inputImage = np.zeros( (200,200,1) ).astype(np.uint8)
      for i,l in enumerate(inds):
        inputImage[l,i]=255

      if source not in imageCache:
        imageCache[source] = cv2.imread(source)

      simg = imageCache[source]
      simg = cv2.cvtColor( simg, cv2.COLOR_BGRA2BGR)
      simg = (cv2.normalize(simg, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)*255.0).astype(np.uint8)
      simg = cv2.resize(simg,(generator.output.shape[2],generator.output.shape[1]),interpolation=cv2.INTER_CUBIC)

      yield inputImage,np.expand_dims(cv2.cvtColor(simg,cv2.COLOR_BGR2GRAY),-1)


pg = genPairs()
x,y = next(pg)

sampleSize=16
n=0

while 1:
  n+=1
  xl,yl = [],[]
  for _ in range(sampleSize):
    x,y = next(pg)
    xl.append(x)
    yl.append(y)

  xo,yo = xl,yl
  pred = generator.predict(np.array(xl)/255.0)

  lossf = discriminator.train_on_batch([np.array(xl)/255.0,np.array(pred)],      np.zeros( (sampleSize,) )*1.0 )
  losst = discriminator.train_on_batch([np.array(xl)/255.0,np.array(yl)/255.0],  np.ones(  (sampleSize,) )*1.0 )

  xl,yl = [],[]
  for _ in range(sampleSize):
    x,y = next(pg)
    xl.append(x)
    yl.append(y)

  prevcombdloss  = combined.train_on_batch([np.array(xl)/255.0], [np.ones(  (sampleSize,) )*1.0 ])
  print(lossf,losst,prevcombdloss)


  cv2.imshow('pred', np.vstack([yo[0],np.clip(0.0,255.0, pred[0]*255.0 ).astype(np.uint8)]) )          
                                    
  k = cv2.waitKey(1)


  if n>0 and (n%100==0 or k==ord('q')):
    cv2.imwrite('gensample.png' , np.clip(0.0,255.0, pred[0]*255.0 ).astype(np.uint8) )
    print('save_weights')
    generator.save_weights('generator.bin')
    discriminator.save_weights('discriminator.bin')
  if k==ord('q'):
    break