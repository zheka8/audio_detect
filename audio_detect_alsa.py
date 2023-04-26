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
    WINDOW_SIZE_IN_SEC = 10 # a sliding window to use for processing data
    WINDOW_SHIFT_IN_SEC = 1 # slide the window by this amount
    DETECTION_THRESHOLD = 0.7

    # Set up audio input
    input_device = 'hw:2,0' #TO DO: need to set this programatically depending on which card USB audio ends up (see .bashrc).
    input_channels = 1
    input_rate = 44100
    input_format = alsaaudio.PCM_FORMAT_S16_LE
    input_period_size = 1024
    num_periods = int(input_rate/input_period_size)

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
   
    match_audio, match_stds, match_names = read_sound_samples()


    # Initialize circular buffer
    #buffer_size = input_rate * 10  # buffer size of 10 seconds
    buffer_size = num_periods * input_period_size * WINDOW_SIZE_IN_SEC
    window = np.zeros(buffer_size, dtype=np.int16)
    start = 0  # start index of circular buffer

    win_num = 0
    frames_list = []
    while win_num < 10:
        # Read data from capture device
        new_portion = np.zeros((num_periods,input_period_size), dtype=np.int16)
        for i in range(num_periods): #read one second worth of data
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
        start_time = time.time()
        #a = match_audio[0].astype(float)/MAX_VAL
        b = window.astype(float)/MAX_VAL
        std_b = np.std(b)
        #norm_factor = np.std(a) * np.std(b) * len(a)
        #corr_max = np.amax(correlate(a, b, mode='valid')/norm_factor)
       
        # compare window to several possible matches
        for i in range(len(match_audio)):
            a = match_audio[i]
            std_a = match_stds[i]
            norm_factor = std_a * std_b * len(a)
            corr_max = np.amax(correlate(a, b, mode='valid')/norm_factor)
       
            if corr_max > DETECTION_THRESHOLD:
                print(f'DETECTED {match_names[i]}  {corr_max:0.3f}')

        print(f'Processed in {time.time() - start_time}')
    
    filename = 'output' + str('all')
    print('Plotting')
    plot_window(b, filename)
    plot_window(a, 'match')
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
    MAX_VAL = 32767
    folder = 'sounds'
    filenames = ['outlook_email.wav', 'teams_alert.wav', 'teams_incoming_call.wav']
    audio_files = []
    stds = []
    for filename in filenames:
        fs, data = wavfile.read(path.join(folder, filename), 'rb')
        print('fs: ', fs)
        data_float = data[:,0].astype(float)/MAX_VAL
        audio_files.append(data_float)
        stds.append(np.std(data_float))

    return audio_files, stds, filenames


if __name__ == '__main__':
    record_data()
