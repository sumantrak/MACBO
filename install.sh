#!/bin/bash
apt-get update
apt-get install -y python3.8 
apt-get install -y python3-pip
pip install -r requirements.txt
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install -e models/dib
