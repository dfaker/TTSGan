


import glob
import random
import os
import model

os.path.exists('converted') or os.mkdir('converted')

imageDim=128*2
sections=1
gf = 64*2
df = 64*2

generator,discriminator,combined = model.getJoinedModel(upscaleLayers=5)

import cv2
import numpy as np

try:
  generator.load_weights('generator.bin')
except Exception as e:
  print(e)

melmin,melmax,sample_rate = -20.70956,2.0575275,22050

def scale_minmax(X, min=0.0, max=1.0):
  X_std = (X - X.min()) / (X.max() - X.min())
  X_scaled = X_std * (max - min) + min
  return X_scaled

sources= [x for x in  list(glob.glob('sourceSamples\\*.png')) if '\\Zaud' not in x]
random.shuffle(sources)
n_mels = 128
hop_length = 512

import librosa

letters = ' DHPTCIYKBLXRWAQVGZNFSMUEOJ#'

testPhrase = 'TO THIS I ALWAYS ADD SOME NORMAL BREAD FOR BREADCRUMBS'.upper()

name = testPhrase.split('\\')[-1]
name = name.replace('.png','').strip()
print(name)
inds = [letters.index(x) if x in letters else 0 for x in name]

inputImage = np.zeros( (200,200,1) ).astype(np.uint8)
for i,l in enumerate(inds):
  inputImage[l,i]=255

pred = generator.predict(np.array([inputImage])/255.0)

pred = cv2.resize( ( np.clip(0.0,255.0, pred[0]*255.0 ) ).astype(np.uint8), (216,128),interpolation=cv2.INTER_CUBIC)
mels = cv2.resize(pred,(216,128),interpolation=cv2.INTER_CUBIC)
mels = scale_minmax(mels.astype(np.float32),min=melmin,max=melmax)
mels = np.exp(mels)
y = librosa.feature.inverse.mel_to_audio(mels, sr=sample_rate, n_fft=hop_length*2, hop_length=hop_length)
y = librosa.util.normalize(y)

librosa.output.write_wav('converted\\TESTPHRASE.wav', y, sample_rate)

for i,src in enumerate(sources[:10]):

  print(i,src)
  oimg = cv2.imread(src)
  simg = cv2.cvtColor( oimg, cv2.COLOR_BGRA2BGR)
  simg = (cv2.normalize(simg, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)*255.0).astype(np.uint8)
  simg = cv2.resize(simg,(imageDim*2,imageDim),interpolation=cv2.INTER_AREA)

  name = src.split('\\')[-1]
  name = name.replace('.png','').strip().upper().replace('[MUSIC]','#')
  print(name)
  inds = [letters.index(x) if x in letters else 0 for x in name]
  
  print(inds)

  inputImage = np.zeros( (200,200,1) ).astype(np.uint8)
  for i,l in enumerate(inds):
    inputImage[l,i]=255

  print(i,src,'Predict')
  pred = generator.predict(np.array([inputImage])/255.0)
  pred = cv2.resize( ( np.clip(0.0,255.0, pred[0]*255.0 ) ).astype(np.uint8), (216,128),interpolation=cv2.INTER_CUBIC)

  #Convert Predicted
  print(i,src,'Convert Predicted')
  mels = cv2.resize(pred,(216,128),interpolation=cv2.INTER_CUBIC)
  mels = scale_minmax(mels.astype(np.float32),min=melmin,max=melmax)
  mels = np.exp(mels)
  y = librosa.feature.inverse.mel_to_audio(mels, sr=sample_rate, n_fft=hop_length*2, hop_length=hop_length)
  y = librosa.util.normalize(y)
  librosa.output.write_wav('converted\\{}_conv.wav'.format(name), y, sample_rate)
