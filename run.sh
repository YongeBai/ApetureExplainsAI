#!/bin/bash

python3 llm/Wizard-Vicuna-13B-Uncensored-SuperHOT-8K-GPTQ/inference.py --file_path $1
python3 inference.py --file_path $1