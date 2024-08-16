# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 00:02:18 2020

@author: aaa
"""

from flask import Flask
app=Flask(__name__)
@app.route('/')
def home():
    return "Hello! World."

app.run(port=5000)