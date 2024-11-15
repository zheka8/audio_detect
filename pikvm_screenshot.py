#!/usr/bin/python3
import os
import requests
from urllib3.exceptions import InsecureRequestWarning

def take_screenshot():
    # Take a screenshot using PiKVM api. Return the path to screenshot of successful, otherwise return None
    hostname = os.getenv('PIKVM_HOSTNAME')
    username = os.getenv('PIKVM_USER')
    password = os.getenv('PIKVM_PW')

    streamer_api_url = f'https://{hostname}/api/streamer/snapshot'
    info_api_url = f'https://{hostname}/api/info'

    snapshot_path = '/home/pi/Projects/audio_detect'
    snapshot_name = 'screenshot.jpeg'

    requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
    response = requests.get(streamer_api_url, verify=False, auth=(username, password))

    if response.status_code == 200:
        with open(os.path.join(snapshot_path, snapshot_name), 'wb') as f:
            f.write(response.content)


if __name__ == '__main__':
    take_screenshot()