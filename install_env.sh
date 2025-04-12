#!/bin/bash

set -e

pip3 install torch torchvision torchaudio

pip install -U decord


pip install -U git+https://github.com/huggingface/transformers.git

pip install -U deepspeed==0.16.4

pip install -U datasets

pip install -U accelerate==1.4.0

pip install -U flash-attn==2.7.4.post1 --no-build-isolation

pip install qwen-vl-utils



