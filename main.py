# -*- coding: utf-8 -*-
"""
Created on Feb 2021

@author: Natalia
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
import scipy
import csv
import pandas as pd

file = 'Nghi_BYB_Recording_2022-01-10_12.41.30'

file2 = 'Dayanna_BYB_Recording_2022-01-10_10.17.10.wav'

fs, data = waves.read(file)

length_data=np.shape(data)
length_new=length_data[0]*0.05
ld_int=int(length_new)
from scipy import signal
data_new=signal.resample(data,ld_int)

plt.figure('Spectrogram')
d, f, t, im = plt.specgram(data_new, NFFT= 256, Fs=500, noverlap=250)
plt.ylim(0,90)
plt.colorbar(label= "Power/Frequency")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.savefig("Figure 1")

matrixf=np.array(f).T
np.savetxt('Frequencies.csv', matrixf)
df = pd.read_csv("Frequencies.csv", header=None, index_col=None)
df.columns = ["Frequencies"]
df.to_csv("Frequencies.csv", index=False)

position_vector=[]
length_f=np.shape(f)
l_row_f=length_f[0]
for i in range(0, l_row_f):
    if f[i]>=7 and f[i]<=12:
        position_vector.append(i)

length_d=np.shape(d)
l_col_d=length_d[1]
AlphaRange=[]
for i in range(0,l_col_d):
    AlphaRange.append(np.mean(d[position_vector[0]:max(position_vector)+1,i]))


def smoothTriangle(data, degree):
    triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1])) # up then down
    smoothed=[]

    for i in range(degree, len(data) - degree * 2):
        point=data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point)/np.sum(triangle))
    # Handle boundaries
    smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed

plt.figure('AlphaRange')
y=smoothTriangle(AlphaRange, 100)
plt.plot(t, y)
plt.xlabel('Time [s]')
plt.xlim(0,max(t))
plt.savefig("Figure 2")

datosy=np.asarray(y)
datosyt=np.array(
        [
        datosy,
        t
        ])
with open ('datosyt.csv', 'w', newline='') as file:
    writer=csv.writer(file, dialect='excel-tab')
    writer.writerows(datosyt.T)
    

df = pd.read_csv("datosyt.csv", header=None, index_col=None)
df.columns = ["Power                   Time"]
df.to_csv("datosyt.csv", index=False)



tg=np.array([4.2552,14.9426, 23.2801,36.0951, 45.4738,59.3751, 72.0337,85.0831, max(t)+1])

length_t=np.shape(t)
l_row_t=length_t[0]
eyesclosed=[]
eyesopen=[]
j=0  #initial variable to traverse tg
l=0  #initial variable to loop through the "y" data
for i in range(0, l_row_t):
    if t[i]>=tg[j]:
        
        if j%2==0:
            eyesopen.append(np.mean(datosy[l:i]))
        if j%2==1:
            eyesclosed.append(np.mean(datosy[l:i]))
        l=i
        j=j+1

        
plt.figure('DataAnalysis')
plt.boxplot([eyesopen, eyesclosed], sym = 'ko', whis = 1.5)
plt.xticks([1,2], ['Eyes open', 'Eyes closed'], size = 'small', color = 'k')
plt.ylabel('AlphaPower')
plt.savefig("Figure 3")

meanopen=np.mean(eyesopen)
meanclosed=np.mean(eyesclosed)
sdopen=np.std(eyesopen)
sdclosed=np.std(eyesclosed)
eyes=np.array([eyesopen, eyesclosed])

from scipy import stats
result=stats.ttest_ind(eyesopen, eyesclosed, equal_var = False)
print(result)





#PART 2 here
import numpy as np2
import matplotlib.pyplot as plt2
import scipy.io.wavfile as waves2
import scipy
import csv
import pandas as pd2
fs2, data2 = waves2.read(file2)

length_data2=np2.shape(data2)
length_new2=length_data2[0]*0.05
ld_int2=int(length_new2)
from scipy import signal
data_new2=signal.resample(data2,ld_int2)

plt2.figure('Spectrogram2')
d2, f2, t2, im2 = plt2.specgram(data_new2, NFFT= 256, Fs=500, noverlap=250)
plt2.ylim(0,90)
plt2.colorbar(label= "Power/Frequency")
plt2.ylabel('Frequency [Hz]')
plt2.xlabel('Time [s]')
plt2.savefig("Figure 4")

matrixf2=np2.array(f2).T
np2.savetxt('Frequencies2.csv', matrixf2)
df2 = pd2.read_csv("Frequencies2.csv", header=None, index_col=None)
df2.columns = ["Frequencies2"]
df2.to_csv("Frequencies2.csv", index=False)

position_vector2=[]
length_f2=np2.shape(f2)
l_row_f2=length_f2[0]
for i in range(0, l_row_f2):
    if f2[i]>=7 and f2[i]<=12:
        position_vector2.append(i)

length_d2=np2.shape(d2)
l_col_d2=length_d2[1]
AlphaRange2=[]
for i in range(0,l_col_d2):
    AlphaRange2.append(np2.mean(d2[position_vector2[0]:max(position_vector2)+1,i]))


def smoothTriangle2(data2, degree2):
    triangle2=np2.concatenate((np2.arange(degree2 + 1), np2.arange(degree2)[::-1])) # up then down
    smoothed2=[]

    for i in range(degree2, len(data2) - degree2 * 2):
        point2=data2[i:i + len(triangle2)] * triangle2
        smoothed2.append(np2.sum(point2)/np2.sum(triangle2))
    # Handle boundaries
    smoothed2=[smoothed2[0]]*int(degree2 + degree2/2) + smoothed2
    while len(smoothed2) < len(data2):
        smoothed2.append(smoothed2[-1])
    return smoothed2

plt2.figure('AlphaRange2')
y2=smoothTriangle2(AlphaRange2, 100)
plt2.plot(t2, y2)
plt2.xlabel('Time2 [s]')
plt2.xlim(0,max(t2))
plt2.savefig("Figure 5")

datosy2=np2.asarray(y2)
datosyt2=np2.array(
        [
        datosy2,
        t2
        ])
with open ('datosyt2.csv', 'w', newline='') as file2:
    writer2=csv.writer(file2, dialect='excel-tab')
    writer2.writerows(datosyt2.T)
    

df2 = pd2.read_csv("datosyt.csv", header=None, index_col=None)
df2.columns = ["Power                   Time"]
df2.to_csv("datosyt2.csv", index=False)



tg2=np2.array([4.2552,14.9426, 23.2801,36.0951, 45.4738,59.3751, 72.0337,85.0831, max(t2)+1])

length_t2=np2.shape(t2)
l_row_t2=length_t2[0]
eyesclosed=[]
eyesopen=[]
j=0  #initial variable to traverse tg
l=0  #initial variable to loop through the "y" data
for i in range(0, l_row_t2):
    if t2[i]>=tg[j]:
        
        if j%2==0:
            eyesopen.append(np2.mean(datosy[l:i]))
        if j%2==1:
            eyesclosed.append(np2.mean(datosy[l:i]))
        l=i
        j=j+1

        
plt2.figure('DataAnalysis')
plt2.boxplot([eyesopen, eyesclosed], sym = 'ko', whis = 1.5)
plt2.xticks([1,2], ['Eyes open', 'Eyes closed'], size = 'small', color = 'k')
plt2.ylabel('AlphaPower')
plt2.savefig("Figure 6")

meanopen=np2.mean(eyesopen)
meanclosed=np2.mean(eyesclosed)
sdopen=np2.std(eyesopen)
sdclosed=np2.std(eyesclosed)
eyes=np2.array([eyesopen, eyesclosed])

from scipy import stats
result=stats.ttest_ind(eyesopen, eyesclosed, equal_var = False)
print(result)