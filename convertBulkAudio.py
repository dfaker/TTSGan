
import librosa
import cv2
import numpy as np
import os
import glob

def scale_minmax(X, min=0.0, max=1.0):
  X_std = (X - X.min()) / (X.max() - X.min())
  X_scaled = X_std * (max - min) + min
  return X_scaled

n_mels = 128
hop_length = 512

src = 'source.mp3'
destPath = 'sourceSamples'
os.path.exists(destPath) or os.mkdir(destPath)

preemphasis = 0.97

s=0
i=0
mini=0
while 1: 
  i+=1
  if i>=mini and not os.path.exists( os.path.join(destPath, 'Zaud{}.wav').format(str(i)) ):
    audioClip, sample_rate = librosa.load(src,mono=True,sr=22050,offset=s,duration=5)

    audioClip = librosa.util.normalize(audioClip)

    mels = librosa.feature.melspectrogram(y=audioClip, sr=sample_rate, n_mels=n_mels,n_fft=hop_length*2, hop_length=hop_length)
    mels = np.log(mels + 1e-9)

    melmin,melmax = mels.min(),mels.max()

    mels = scale_minmax(mels, 0, 255).astype(np.uint8)

    print(i,mels.shape,'-',mels.mean(),'-',melmin,melmax,sample_rate)
    
    cv2.imwrite( os.path.join(destPath, 'Zaud{}.png').format(str(i)),mels)
    librosa.output.write_wav(os.path.join(destPath, 'Zaud{}.wav').format(str(i)), audioClip, sample_rate)

  s+= 5