#!/usr/bin/python3
import alsaaudio
import wave
import numpy as np
from scipy.signal import correlate
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
from os import path
import time
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import logging
from logging.handlers import RotatingFileHandler

def record_data():
    setup_logging() 

    # some constants
    MAX_VAL = 32767 #16 bit int max for scaling
    WINDOW_SIZE_IN_SEC = 10 # a sliding window to use for processing data
    WINDOW_SHIFT_IN_SEC = 1 # slide the window by this amount
    DETECTION_THRESHOLD = 0.7
    MAG_THRESHOLD = 0.05 * MAX_VAL    # amplitude threshold for initiating processing of each window
    FRAME_PROCESSING_DELAY = 2 # number of frames to keep reading after amplitude detection before processing the window

    # Set up audio input
    audio_card_id = os.environ.get('USB_AUDIO_CARD_ID', 2) # default to card 2
    input_device = f'hw:{audio_card_id},0' #TO DO: need to set this programatically depending on which card USB audio ends up (see .bashrc).
    logging.info(input_device)
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

    match_audio, match_stds, match_names = read_sound_samples()

    # Initialize circular buffer
    #buffer_size = input_rate * 10  # buffer size of 10 seconds
    buffer_size = num_periods * input_period_size * WINDOW_SIZE_IN_SEC
    window = np.zeros(buffer_size, dtype=np.int16)
    start = 0  # start index of circular buffer

    win_num = 0
    frames_list = []
    process_frame_counter = FRAME_PROCESSING_DELAY
    start_countdown = False
    while win_num < 40000:
        # Read data from capture device
        new_portion = np.zeros((num_periods,input_period_size), dtype=np.int16)
        for i in range(num_periods): #read one second worth of data
            length, data = input_stream.read()
            
            # skip this frame if something went wrong and there's nothing captured
            if length != input_period_size:
                continue

            new_portion[i,:] = np.frombuffer(data, dtype=np.int16)
            frames_list.append(data)          

        logging.debug(f'Window {win_num}    Process Frame Counter {process_frame_counter}')

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
        
        # Process the latest window of data if magnitude is above threshold
        start_time = time.time()
        
        # only perform processing if the counter hits zero
        if process_frame_counter <= 0:
            # current window audio
            b = window.astype(float)/MAX_VAL        
            std_b = np.std(b)
       
            # compare window to several possible matches
            for i in range(len(match_audio)):
                a = match_audio[i]
                std_a = match_stds[i]
                norm_factor = std_a * std_b * len(a)
                corr_start_time = time.time()
                corr_max = np.amax(correlate(a, b, mode='valid')/norm_factor)
                logging.debug(f'Corr exec time: {time.time() - corr_start_time:0.4f}')

                # handle detection (only handle first match to save processing time)
                if corr_max > DETECTION_THRESHOLD:
                    message_text = f'DETECTED {match_names[i]}  {corr_max:0.3f}'
                    logging.debug(message_text)
                    notify_by_slack('C0517S2GUBY', os.environ['SLACK_API_TOKEN'], message_text)
                    break
            
            # reset processing flag
            process_frame_counter = FRAME_PROCESSING_DELAY
            start_countdown = False
        
        logging.debug(f'Processed in {time.time() - start_time}')
         
        # set a counter to delay processing if current frame exceeded amplitude threshold
        # this allows some more data into the window for better detections if the sound spans across frames
        if np.amax(np.abs(new_portion)) > MAG_THRESHOLD:
            process_frame_counter = FRAME_PROCESSING_DELAY
            start_countdown = True
        
        if start_countdown:
            process_frame_counter -= 1
    
    #filename = 'output' + str('all')
    #print('Plotting')
    #plot_window(b, filename)
    #plot_window(a, 'match')
    #print('Saving wav')
    #save_wav(frames_list, filename, input_channels, input_rate)
    
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
    folder = '/home/pi/Projects/audio_detect/sounds'
    filenames = ['outlook_email_short.wav', 'teams_alert.wav', 'teams_incoming_call_short.wav']
    audio_files = []
    stds = []
    for filename in filenames:
        fs, data = wavfile.read(path.join(folder, filename), 'rb')
        print('fs: ', fs)
        data_float = data[:,0].astype(float)/MAX_VAL
        audio_files.append(data_float)
        stds.append(np.std(data_float))

    return audio_files, stds, filenames

def notify_by_slack(channel_id, slack_api_token, message_text):
    # Set the text of the message you want to send
    client = WebClient(token=slack_api_token)

    try:
        # Call the chat.postMessage method using the WebClient
        response = client.chat_postMessage(
            channel=channel_id,
            text=message_text
        )
        print("Message sent: ", response["ts"])
    except SlackApiError as e:
        print("Error sending message: {}".format(e))


def setup_logging():
    # Configure the logging system
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            RotatingFileHandler('/home/pi/Projects/audio_detect/logs/audio_detect.log', maxBytes=1024*1024, backupCount=3),
            logging.StreamHandler()
        ]
    )


if __name__ == '__main__':
    record_data()
