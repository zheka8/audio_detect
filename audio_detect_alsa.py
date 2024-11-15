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
import requests
from urllib3.exceptions import InsecureRequestWarning

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
    new_portion = np.zeros((num_periods,input_period_size), dtype=np.int16)
    length, data = 0, 0
    b = np.zeros_like(window)
    std_b = 0
    start_countdown = False

    # Start the memory trace 
    # tracemalloc.start()

    while win_num < 40 or True:
        # Read data from capture device
        for i in range(num_periods): #read one second worth of data
            length, data = input_stream.read()
            
            # skip this frame if something went wrong and there's nothing captured
            if length != input_period_size:
                continue

            new_portion[i,:] = np.frombuffer(data, dtype=np.int16)
            #frames_list.append(data)       # this is a memory leak. use a circular buffer or queue instead   

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

        logging.debug(f'Window {win_num}')
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
                    message_text = f'{match_names[i]}'
                    logging.info(message_text)
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
   
    """
    # Memory profile stats
    snapshot = tracemalloc.take_snapshot()
    #print(f"Current memory usage: {current / 10**6}MB")

    top_stats = snapshot.statistics('lineno')
    print("Top 10 memory-consuming lines:")
    for stat in top_stats[:10]:
        print(stat)    
    """

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
        logging.info(f'Reading {filename} fs: {fs}')
        data_float = data[:,0].astype(float)/MAX_VAL
        audio_files.append(data_float)
        stds.append(np.std(data_float))

    return audio_files, stds, filenames

def notify_by_slack(channel_id, slack_api_token, message_text):
    # Set the text of the message you want to send
    client = WebClient(token=slack_api_token)

    image_path = take_screenshot()

    # use a placeholder image to prevent slack API from failing in case the screenshot failed
    if not image_path:
        image_path = '/home/pi/Projects/audio_detect/placeholder.jpeg'

    try:
        # Upload an image
        response = client.files_upload(
            channels = channel_id,
            file = image_path,
            title = message_text,
            initial_comment = message_text            
        )
        logging.info(f'Message sent')
    except SlackApiError as e:
        logging.info(f'Error sending message: {e}')


def setup_logging():
    # Configure the logging system
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            RotatingFileHandler('/home/pi/Projects/audio_detect/logs/audio_detect.log', maxBytes=1024*1024, backupCount=3),
            logging.StreamHandler()
        ]
    )


def take_screenshot():
    # Take a screenshot using PiKVM api. Return the path to screenshot of successful, otherwise return None

    hostname = os.getenv('PIKVM_HOSTNAME')
    username = os.getenv('PIKVM_USER')
    password = os.getenv('PIKVM_PW')

    streamer_api_url = f'https://{hostname}/api/streamer/snapshot'
    #info_api_url = f'https://{hostname}/api/info'

    snapshot_path = '/home/pi/Projects/audio_detect'
    snapshot_name = 'screenshot.jpeg'

    requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
    logging.info(f'PiKVM API call to {streamer_api_url}')

    try:
        response = requests.get(streamer_api_url, verify=False, auth=(username, password))
        if response.status_code == 200:
            logging.info(f'PiKVM screenshot succeeded with status code {response.status_code}')
            with open(os.path.join(snapshot_path, snapshot_name), 'wb') as f:
                f.write(response.content)
                return os.path.join(snapshot_path, snapshot_name)
        else:
            logging.info(f'PiKVM screenshot failed with status code {response.status_code}')
            return ''
    except Exception as e:
        logging.info(f'PiKVM screenshot attempted and caught exception {e}')
        return ''


if __name__ == '__main__':
    #tracemalloc.start()

    record_data()
    '''
    # Print the current memory usage
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 10**6}MB")

    # Stop the trace and print the top 10 memory-consuming lines of code
    tracemalloc.stop()
    top_stats = tracemalloc.get_stats()[:10]
    print("Top 10 memory-consuming lines:")
    for stat in top_stats:
        print(stat)
    '''
