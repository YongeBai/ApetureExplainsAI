#!/bin/bash

python3 llm_inference.py --file_path $1
python3 tts_inference.py --file_path $1