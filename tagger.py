
import tkinter as tk
import os
import mpv

p = mpv.MPV()
p.volume=50
p.loop='inf'
names = []
sampleFolder = 'sourceSamples'

blocksize=1024*10

textSource = open('transcript.txt','r')

for fn in os.listdir(sampleFolder):
  if '.wav' in fn:
    base =  os.path.join(sampleFolder,fn.replace('.wav','')) 
    if os.path.exists(base+'.png'):
      names.append(base)
      print(base)

names = sorted(names,reverse=True,key=lambda x:int(x.replace(sampleFolder+'\\Zaud','')))

root = tk.Tk()

currentFile = None
txt = tk.Text(root, height=50, width=80)

def cmdassign():
  global currentFile
  print('assign')
  try:
    selected = txt.get(tk.SEL_FIRST, tk.SEL_LAST)
    txt.delete(1.0, tk.SEL_FIRST)
    os.remove(currentFile+'.wav')
    os.rename(currentFile+'.png', os.path.join( sampleFolder,selected+'.png') )
  except Exception as e:
    print(e)
  if len(txt.get('0.0',tk.END))<blocksize:
    txt.insert(tk.END, textSource.read(blocksize) )
  currentFile = names.pop()
  print(currentFile)
  p.play(currentFile+'.wav')

def cmdskip():
  global currentFile
  if currentFile is not None:
    os.remove(currentFile+'.wav')
    os.remove(currentFile+'.png')
  print(len(txt.get('0.0',tk.END)))
  if len(txt.get('0.0',tk.END))<blocksize:
    txt.insert(tk.END, textSource.read(blocksize) )

  currentFile = names.pop()
  print(currentFile)
  p.play(currentFile+'.wav')

assign = tk.Button(root,text='assign',command=cmdassign)
assign.pack()
skip = tk.Button(root,text='skip',command=cmdskip)
skip.pack()

txt.pack()
txt.insert(tk.END, textSource.read(blocksize) )

root.mainloop()