#!/bin/bash
pip3 install yapf --upgrade
python3 -m yapf -ir -vv --style ./.style.yapf verl tests single_controller examples recipe
