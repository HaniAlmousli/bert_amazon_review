#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""

@author: hanialmousli
"""

import os
import json
import pickle
import io
import sys
import signal
import traceback
from flask import Flask,jsonify
import flask
import tensorflow as tf
import pandas as pd
import sys
import numpy as np
sys.path.append('..')

import helper 
from SentimentPredictor import SentimentPredictor

import pdb

model = SentimentPredictor()
optimizer = helper.get_optimizer()

checkpoint_path ="../../local_test/test_dir/model/checkpoints/train"
ckpt = tf.train.Checkpoint(classifier=model,
                                optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)
ckpt.restore(ckpt_manager.latest_checkpoint)

app = Flask(__name__)
# swagger = Swagger(app)


@app.route('/predict', methods=["POST"])
def predict():
    """Example file endpoint returning a prediction of iris
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
    """
    data = flask.request.data.decode('utf-8')
    reviews_txt = io.StringIO(data)
    lst_input = reviews_txt.getvalue().split('\n')
    res = tf.nn.softmax(model(lst_input),axis=1)
    print("\n\n")
    print(res)
    print("\n\n")
    

    out = io.StringIO()
    np.savetxt(out,res.numpy(),fmt='%1.3f')
    
    return flask.Response(response=out.getvalue(), status=200, mimetype='text/csv')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)