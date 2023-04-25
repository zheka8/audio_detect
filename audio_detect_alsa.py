#!/usr/bin/python3
import alsaaudio
import wave
import numpy as np
from scipy.signal import correlate
from scipy.io import wavfile
import matplotlib.pyplot as plt
from os import path
import time

def record_data():
    # some constants
    MAX_VAL = 32767 #16 bit int max for scaling
    WINDOW_SIZE_IN_SEC = 5 # a sliding window to use for processing data
    WINDOW_SHIFT_IN_SEC = 1 # slide the window by this amount

    # Set up audio input
    input_device = 'hw:2,0' #TO DO: need to set this programatically depending on which card USB audio ends up (see .bashrc).
    input_channels = 1
    input_rate = 44100
    input_format = alsaaudio.PCM_FORMAT_S16_LE
    input_period_size = 1024

    input_stream = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, input_device)
    input_stream.setchannels(input_channels)
    input_stream.setrate(input_rate)
    input_stream.setformat(input_format)
    input_stream.setperiodsize(input_period_size)

    # Record audio
    '''
    frames = []
    for i in range(0, int(input_rate / input_period_size * 10)):
        # Read data from the input stream
        data_length, data = input_stream.read() # returns a tuple (size, bytes)
        frames.append(data)
    '''
   
    match_audio, match_audio_names = read_sound_samples()


    # Initialize circular buffer
    #buffer_size = input_rate * 10  # buffer size of 10 seconds
    buffer_size = 43 * 1024 * 10
    window = np.zeros(buffer_size, dtype=np.int16)
    start = 0  # start index of circular buffer

    win_num = 0
    frames_list = []
    while win_num < 10:
        # Read data from capture device
        new_portion = np.zeros((43,1024), dtype=np.int16)
        for i in range(43): #read one second worth of data
            length, data = input_stream.read()
            new_portion[i,:] = np.frombuffer(data, dtype=np.int16)
            frames_list.append(data)          

        print(win_num)

        #frames = np.frombuffer(data, dtype=np.int16)
        frames = new_portion.ravel()

        # Add frames to circular buffer
        end = start + len(frames)
        if end > buffer_size:
            # Wrap around to the start of the buffer
            window[start:] = frames[:buffer_size-start]
            window[:end-buffer_size] = frames[buffer_size-start:]
            start = end - buffer_size
        else:
            window[start:end] = frames
            start = end

        win_num += 1
        
        # Process the latest window of data
        #time.sleep(5)
        print(match_audio[0].shape)
        print(window.shape)
        norm_factor = np.sqrt(np.sum((match_audio[0]/MAX_VAL)**2) * np.sum((window/MAX_VAL)**2))
        corr_max = np.amax(correlate(match_audio[0], window, mode='same')/norm_factor)
        print(corr_max)

    
    filename = 'output' + str('all')
    print('Plotting')
    plot_window(new_portion.ravel()/MAX_VAL, filename)
    print('Saving wav')
    save_wav(frames_list, filename, input_channels, input_rate)
    
    # Clean up
    input_stream.close()


def save_wav(frames, filename, channels, rate):    
    """ save a list of frames as a wav file """
    # Save to WAV file
    output_file = wave.open(filename + '.wav', 'wb')
    output_file.setnchannels(channels)
    output_file.setsampwidth(2) #16 bit sample width
    output_file.setframerate(rate)
    output_file.writeframes(b''.join(frames))
    output_file.close()

def plot_window(window, filename):
    """ generate plot """
    plt.plot(window)
    plt.savefig(filename + '.png')

def read_sound_samples():
    """ read in the samples used for matching and return them as a tuple 
    only use the first channel
    """
    folder = 'sounds'
    filenames = ['outlook_email.wav', 'teams_alert.wav', 'teams_incoming_call.wav']
    audio_files = []
    for filename in filenames:
        fs, data = wavfile.read(path.join(folder, filename), 'rb')
        audio_files.append(data[:,0])

    return audio_files, filenames


if __name__ == '__main__':
    record_data()
