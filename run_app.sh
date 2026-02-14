#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/.local/lib/python3.11/site-packages
/opt/anaconda3/bin/python3 -m streamlit run app.py
