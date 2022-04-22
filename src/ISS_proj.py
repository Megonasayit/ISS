import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import cmath
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk

def dft(frame, lenght):
    t = np.zeros(lenght, dtype = 'complex_')
    N = len(frame)
    while(N < lenght):
        N +=1
        frame = np.append(frame, 0)
    
    for k in range(N):
        for n in range(N):
            t[k] += frame[n]*cmath.exp(-2j*cmath.pi*k*n*(1/N))
    return t

def idft(frame, lenght):
    x = np.zeros(lenght, dtype = 'complex_')
    N = len(frame)
    while(N < lenght):
        N +=1
        frame = np.append(frame, 0)

    for n in range(N):
        for k in range(N):
            x[n] += frame[k]*cmath.exp(2j*cmath.pi*k*n*(1/N))
        x[n] /= N
    return x

#------------------uloha3------------------#

print('Uloha 3:')
tone1, tone1_samplerate = sf.read('../audio/maskoff_tone.wav', start=3*16000, stop=4*16000)
tone2, tone2_samplerate = sf.read('../audio/maskon_tone.wav', start=80025, stop=96025)

tone1 -= np.mean(tone1)
tone2 -= np.mean(tone2)

i = 0
frames_tone1 = np.zeros((99,320))
frames_tone2 = np.zeros((99,320))
while i < 99:
    frames_tone1[i] = tone1[i*160:i*160+320]
    frames_tone2[i] = tone2[i*160:i*160+320]
    i += 1

start1 = 0 * 0.02
segment = 0.02

start_frame1 = int(start1 * tone1_samplerate)
end_frame1 = int((start1 + segment) * tone1_samplerate)
tone1_seg = tone1[start_frame1:end_frame1]

plt.figure(figsize=(6,3))
plt.plot(np.arange(tone1_seg.size)/tone1_samplerate + start1, frames_tone1[0], label = 'bez rúška')
plt.plot(np.arange(tone1_seg.size)/tone1_samplerate + start1, frames_tone2[0], label = 's rúškom')
plt.gca().set_xlabel('čas[s]')
plt.gca().set_title('Zvukový signál frame 0')
plt.grid(alpha = 0.5, linestyle = '--')

plt.legend()
plt.tight_layout()
plt.savefig('uloha3.png', dpi=300)

#------------------uloha4------------------#

frames_tone1_clipping = np.copy(frames_tone1)
frames_tone2_clipping = np.copy(frames_tone2)

max1_i = np.unravel_index(np.argmax(np.abs(frames_tone1_clipping), axis = None), np.abs(frames_tone1_clipping).shape)
max2_i = np.unravel_index(np.argmax(np.abs(frames_tone2_clipping), axis = None), np.abs(frames_tone2_clipping).shape)

max1 = np.abs(frames_tone1[max1_i]*0.7)
max2 = np.abs(frames_tone2[max2_i]*0.7)

i = 0
while i < 99:
    for x in frames_tone1_clipping[i]:
        if np.any(x > max1):
            frames_tone1_clipping[i, np.where(frames_tone1_clipping[i]==x)] = 1
        elif np.any(x < -max1):
            frames_tone1_clipping[i, np.where(frames_tone1_clipping[i]==x)] = -1
        else:
            frames_tone1_clipping[i, np.where(frames_tone1_clipping[i]==x)] = 0
        
    for y in frames_tone2_clipping[i]:
        if np.any(y > max2):
            frames_tone2_clipping[i, np.where(frames_tone2_clipping[i]==y)] = 1
        elif np.any(y < -max2):
            frames_tone2_clipping[i, np.where(frames_tone2_clipping[i]==y)] = -1
        else:
            frames_tone2_clipping[i, np.where(frames_tone2_clipping[i]==y)] = 0
    i += 1

correlation_list1 = np.zeros((99,320))
correlation_list2 = np.zeros((99,320))
lag1 = np.zeros(99)
lag2 = np.zeros(99)

prah = 25

i = 0
while i < 99:
    k = 0
    while k < 320:
        y = 0
        tmp1 = 0
        tmp2 = 0
        while y < 319 - k:
            tmp1 += frames_tone1_clipping[i, y] * frames_tone1_clipping[i, y + k]
            tmp2 += frames_tone2_clipping[i, y] * frames_tone2_clipping[i, y + k]
            y += 1
        correlation_list1[i, k] = tmp1
        correlation_list2[i, k] = tmp2
        k += 1
    lag1[i] = (np.max(np.argmax(correlation_list1[i, prah:], axis = None)+prah))
    lag2[i] = (np.max(np.argmax(correlation_list2[i, prah:], axis = None)+prah))
    i += 1
    
print("Uloha 4:")
print("stredna hodnota maskoff:" , np.mean(tone1_samplerate/lag1))
print("stredna hodnota maskon:" , np.mean(tone2_samplerate/lag2))

print("rozptyl maskoff:" , np.var(tone1_samplerate/lag1))
print("rozptyl maskon:" , np.var(tone2_samplerate/lag2))

_,bx = plt.subplots(4, 1,figsize = (16,9))

start3 = 95 * 0.02
segment = 0.02

start_frame1 = int(start1 * tone1_samplerate)
end_frame1 = int((start1 + segment) * tone1_samplerate)
tone1_seg = tone1[start_frame1:end_frame1]

bx[0].plot(np.arange(tone1_seg.size)/tone1_samplerate + start3, frames_tone1[95])
bx[0].set_xlabel('t[s]')
bx[0].set_title('Rámec')
bx[0].grid(alpha = 0.5, linestyle = '--')

bx[1].plot(np.arange(frames_tone1_clipping[95].size), frames_tone1_clipping[95])
bx[1].set_xlabel('vzorky')
bx[1].set_title('Centrálne klipovanie s 70%.')
bx[1].grid(alpha = 0.5, linestyle = '--')

bx[2].plot(np.arange(correlation_list1[95].size), correlation_list1[95])
bx[2].stem([lag1[95]], [correlation_list1[95, int(np.around(lag1[95]))]], linefmt='red', label="Lag")
bx[2].axvline(x=prah, color = 'black', label="Prah")
bx[2].set_xlabel('vzorky')
bx[2].set_title('Autokorelácia.')
bx[2].grid(alpha = 0.5, linestyle = '--')
bx[2].legend()

bx[3].plot(np.arange(99), tone1_samplerate/lag1, label='bez rúška')
bx[3].plot(np.arange(99), tone2_samplerate/lag2, label='s rúškom')
bx[3].set_xlabel('rámce')
bx[3].set_title('Základná frekvencia rámcov.')
bx[3].legend(loc='upper right')
bx[3].grid(alpha = 0.5, linestyle = '--')

plt.tight_layout()
plt.savefig('uloha4.png', dpi=300)

#------------------uloha5------------------#
print('Uloha 5:')
dft_spektrum1 = np.zeros((99,1024), dtype = 'complex_')
dft_spektrum2 = np.zeros((99,1024), dtype = 'complex_')

dft_spektrum1 = (np.fft.fft(frames_tone1, 1024))
dft_spektrum2 = (np.fft.fft(frames_tone2, 1024))

sgr_log1 = np.zeros((99,512))
sgr_log2 = np.zeros((99,512))

for i in range (99):
    for x in range (512):
        sgr_log1[i,x] = 10*np.log10(np.square(np.abs(dft_spektrum1[i,x]))+1e-20)
        sgr_log2[i,x] = 10*np.log10(np.square(np.abs(dft_spektrum2[i,x]))+1e-20)

plt.figure(figsize=(8,3))
plt.imshow(np.transpose(sgr_log1[:,:512]),aspect='auto', origin='lower', extent=(0, 1, 0, 8000))

plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvencia [Hz]')
plt.gca().set_title('Spectrogram bez rúška')
cbar = plt.colorbar()
cbar.set_label('Spektralna hustota', rotation=270, labelpad=15)
plt.tight_layout()
plt.savefig('uloha5_maskoff.png', dpi=300)

plt.imshow(np.transpose(sgr_log2[:,:512]),aspect='auto', origin='lower', extent=(0, 1, 0, 8000))
plt.gca().set_title('Spectrogram s rúškom')
plt.savefig('uloha5_maskon.png', dpi=300)

#------------------uloha6------------------#
print('Uloha 6:')
f_char1 = np.zeros((99,512))
f_char_avg = np.zeros(512)

for i in range (99):
    for x in range(512):
        f_char1[i,x] = np.abs(dft_spektrum2[i,x]/dft_spektrum1[i,x])

f_char_log = 10*np.log10(np.square(np.abs(f_char1))+1e-20)

for i in range(99):
    for x in range(512):
        f_char_avg[x] += f_char_log[i,x]/99

frange = np.arange(f_char_avg.size) /   1000 * 16000
plt.figure()
plt.plot(frange[:frange.size], f_char_avg[:f_char_avg.size])
plt.gca().set_title('Frekvenčná charakteristika rúška')
plt.gca().set_xlabel('Frekvencia [Hz]')
plt.tight_layout()
plt.savefig('uloha6.png', dpi=300)

#------------------uloha7------------------#
print('Uloha 7:')
i_odz = np.zeros(1024, dtype='complex_')
i_odz = np.fft.ifft(f_char_avg, 1024)
i_odz = i_odz[:512]
plt.figure()
plt.plot(i_odz.real)
plt.gca().set_title('Impulzná odozva rúška')
plt.savefig('uloha7.png', dpi=300)

#------------------uloha8------------------#
print('Uloha 8:')
sentence, sentence_samplerate = sf.read('../audio/maskoff_sentence.wav')
sentence2, sentence2_samplerate = sf.read('../audio/maskon_sentence.wav')
tone, tone_samplerate = sf.read('../audio/maskoff_tone.wav')

sentence_filtered = lfilter(i_odz, 1, sentence)
tone_filtered = lfilter(i_odz, 1, tone)

plt.figure()
_,cx = plt.subplots(3, 1, figsize = (16,9))
cx[0].plot(sentence, label = 'veta bez rúška')
cx[0].plot(sentence_filtered.real, label = 'filtrovaná veta so simulovaným rúškom')
cx[0].legend()

cx[1].plot(sentence, label = 'veta bez rúška')
cx[1].plot(sentence2, label = 'veta s rúškom')
cx[1].legend()

cx[2].plot(sentence2, label = 'veta s rúškom')
cx[2].plot(sentence_filtered.real, label = 'filtrovaná veta so simulovaným rúškom')
cx[2].legend()

plt.tight_layout()
sf.write('../audio/sim_maskon_sentence.wav', sentence_filtered.real, 16000)
sf.write('../audio/sim_maskon_tone.wav', tone_filtered.real, 16000)
plt.savefig('uloha8.png', dpi=300)

#------------------uloha11------------------#
print('Uloha 11:')
blackman = np.blackman(320)
frame_window1 = frames_tone1*blackman
frame_window2 = frames_tone2*blackman

dft_frame_default = np.fft.fft(frames_tone1, 1024)
dft_frame_window1 = np.fft.fft(frame_window1, 1024)
dft_frame_window2 = np.fft.fft(frame_window2, 1024)
dft_blackman_window  = np.fft.fft(blackman, 1024)
dft_blackman_window_log = 10*np.log10(np.square(np.abs(dft_blackman_window))+1e-20)

window_char = np.zeros((99,512))
for x in range(99):
    for i in range(512):
        window_char[x,i] = np.abs(dft_frame_window2[x,i]/dft_frame_window1[x,i])

window_char_log = 10*np.log10(np.square(np.abs(f_char1))+1e-20)

window_char_avg = np.zeros(512)
for i in range(99):
    for x in range(512):
        window_char_avg[x] += f_char_log[i,x]/99

window_odz = np.fft.ifft(window_char_avg,1024)
window_filtered = lfilter(window_odz[:512],1,sentence)
window_tone_filtered = lfilter(window_odz[:512],1,tone)
sf.write('../audio/sim_maskon_sentence_window.wav', window_filtered.real, 16000)
sf.write('../audio/sim_maskon_tone_window.wav', window_tone_filtered.real, 16000)

_, dx = plt.subplots(3,1, figsize = (16,9))
dx[0].plot(blackman)
dx[0].set_title('Časová oblasť')

dx[1].plot(dft_blackman_window_log[:512])   
dx[1].set_title('Spektrálna oblasť')
dx[2].plot(dft_frame_default[0,:512].real, label = 'Bez okienkovej funcie')
dx[2].plot(dft_frame_window1[0,:512].real, label = 'Po okienkovej funkcií')
dx[2].set_title('Porovnanie spektra bez a po aplikácií okienkovej funkcie')
dx[2].legend()

plt.tight_layout()
plt.savefig('uloha11.png', dpi=300)

#------------------uloha13------------------#
print('Uloha 13:')

cnt = 0
for x in range(99):
    if(tone1_samplerate/lag1[x] == tone2_samplerate/lag2[x]):
        cnt += 1

frame_tone1_new = np.zeros((cnt,320))
frame_tone2_new = np.zeros((cnt,320))

cnt = 0
for x in range(99):
    if(tone1_samplerate/lag1[x] == tone2_samplerate/lag2[x]):
        frame_tone1_new[cnt] = frames_tone1[x]
        frame_tone2_new[cnt] = frames_tone2[x]
        cnt += 1

dft_new1 = np.fft.fft(frame_tone1_new,1024)
dft_new2 = np.fft.fft(frame_tone2_new,1024)

new_char = np.zeros((len(frame_tone1_new),512))
for x in range(len(frame_tone1_new)):
    for i in range(512):
        new_char[x,i] = np.abs(dft_new2[x,i]/dft_new1[x,i])

new_char_log = 10*np.log10(np.square(np.abs(new_char))+1e-20)

new_char_avg = np.zeros(512)
for i in range(len(frame_tone1_new)):
    for x in range(512):
        new_char_avg[x] += new_char_log[i,x]/99


plt.figure(figsize = (16,9))
new_frange = np.arange(new_char_avg.size)/1000 * 16000
plt.plot(frange[:frange.size], f_char_avg[:f_char_avg.size], label = 'Default')
plt.plot(new_frange[:new_frange.size], new_char_avg[:new_char_avg.size], label = 'Len s rovnakými rámcami')
plt.gca().set_title('Porovnanie frekvenčných charakteristík rúška bez a s použitím len rámcov s rovnakou frekvenciou')
plt.gca().set_xlabel('Frekvencia [Hz]')
plt.legend()
plt.tight_layout()
plt.savefig('uloha13.png', dpi=300)

new_char_odz = np.fft.ifft(new_char_avg,1024)
new_char_filtered = lfilter(new_char_odz[:512],1,sentence)
new_char_filtered_tone = lfilter(new_char_odz[:512],1,tone)
sf.write('../audio/sim_maskon_sentence_only_match.wav', new_char_filtered.real, 16000)
sf.write('../audio/sim_maskon_tone_only_match.wav', new_char_filtered_tone.real, 16000)