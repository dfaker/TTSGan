# TTSGan

Doesn't currently produce anything more than stylized gibberish.

Expects the inputs of `source.mp3` as a source audio recording and `transcript.txt` as the transcipt of that recording.

`convertBulkAudio.py` will split the recoding into 5 second chunks and save them as pairs of wav files and mel spectrograms.

`tagger.py` will play the recordings in order and allow the user to hilight a section of the transcript, on clicking assign the png will be renamed to match the selected section of text.

`train.py` uses these renamed pngs to train a GAN that reproduces the audio from a one hot encoding of the selected transcription section.

`convert.py` feeds the one hot encodings back into the saved network to recover the original audio.
