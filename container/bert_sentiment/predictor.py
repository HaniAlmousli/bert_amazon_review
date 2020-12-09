# This is the file that implements a flask server to do inferences. 

from __future__ import print_function

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
import numpy as np
import helper 
from SentimentPredictor import SentimentPredictor
import pdb

# prefix = '/home/hani/Repos/bert_amazon_review/container/local_test/test_dir/' 
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')


class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """
        Get the model object for this instance, loading it if it's not already loaded. ClassMethod is used
        to load the model only for one time
        """
        if cls.model == None:
            print("LOADING MODEL .....")
            cls.model = SentimentPredictor()
            optimizer = helper.get_optimizer()

            checkpoint_path =os.path.join(model_path,"checkpoints/train")
            ckpt = tf.train.Checkpoint(classifier=cls.model,
                                         optimizer=optimizer)
            ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)
            ckpt.restore(ckpt_manager.latest_checkpoint)

        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a movie id): The data on which to do the predictions (find similarity). """
        
        model = cls.get_model()
        print("Model Loaded....")
        return model(input)

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to numpy array (ex: [[100,250]]) and then convert the predictions back to CSV from Pandas (which really
    just means one dataframe prediction per id. They ll be all joined together
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        reviews_txt = io.StringIO(data)
        lst_input = reviews_txt.getvalue().split('\n')
         
    else:
        print(">>>> ERROR:The request is {}".format(flask.request))
        return flask.Response(response='This predictor only supports JSON data', status=415, mimetype='text/plain')

    print('Invoking with {} records'.format(len(lst_input)))

    # Do the prediction 
    res = tf.nn.softmax(ScoringService.predict(lst_input),axis=1)
    # Convert from numpy back to CSV
    out = io.StringIO()
    np.savetxt(out,res.numpy(),fmt='%1.3f')
    
    return flask.Response(response=out.getvalue(), status=200, mimetype='text/csv')
