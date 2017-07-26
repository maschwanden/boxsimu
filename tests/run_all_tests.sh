#!/bin/bash
source /home/aschi/Envs/boxsimu/bin/activate
python -m unittest discover -s . -p 'test_*.py'
