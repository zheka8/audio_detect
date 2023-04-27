#!/bin/bash

export USB_AUDIO_CARD_ID=$(grep -m 1 -i "USB-Audio" /proc/asound/cards | awk '{print $1}')
export SLACK_API_TOKEN=$(cat /home/pi/slack_api_token.txt)

/usr/bin/python3 /home/pi/Projects/audio_detect/audio_detect_alsa.py
