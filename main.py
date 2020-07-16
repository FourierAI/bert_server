#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: main.py
# @time: 2020-07-14 18:51
# @desc:
from flask import Flask
from flask import jsonify as json
from flask import json as js
from flask import request
from bert_server import MyBertServer
from bert_client import MyBertClient

app = Flask(__name__)
server = MyBertServer()
client = MyBertClient()


@app.route('/')
def index():
    return 'Hello World'


@app.route('/startbert')
def start_bert():
    server.run()
    return 'The sever has been started!'


@app.route('/shutdownbert')
def shutdown():
    server.shutdown()
    return 'The sever has been shutdown!'


@app.route('/simility/', methods=['GET', 'POST'])
def simility():
    if request.method == 'POST':
        json_data = js.loads(request.get_data(as_text=True))
        sent1 = json_data.get('sent1')
        sent2 = json_data.get('sent2')
        return json(client.query_simility_sentence_pair(sent1, sent2))
    else:
        return 'Error, please post json!'

@app.route('/vec/', methods=['GET', 'POST'])
def get_sent_vec():
    if request.method == "POST":
        json_data = js.loads(request.get_data(as_text=True))
        sent = json_data.get('sent')

        return json(client.query_sentence_vec(sent))
    else:
        return 'Error, please post json!'


if __name__ == '__main__':
    # start flask server
    app.run()
