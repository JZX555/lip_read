from werkzeug.utils import secure_filename
from flask import Flask, render_template, jsonify, request, make_response, send_from_directory, abort
from flask_cors import CORS
import time
import json
import os
import base64
import cv2
import numpy as np
 
app = Flask(__name__)
CORS(app, supports_credentials=True)
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])

video = None
cnt = 0
FLAG = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload')
def upload_test():
    return render_template('up.html')
 

# json
@app.route('/up_photo', methods=['POST', 'GET'], strict_slashes=False)
def upload():
    global cnt, FLAG, video
    if(not os.path.exists('./photo/')):
        os.makedirs('./photo/')

    if(FLAG == False):
        FLAG = True
        cnt = 0
        video = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'XVID'), 20.0, (720,480))

    img = request.values['photo']
    img = img.replace('data:image/png;base64,', '')
    img = base64.b64decode(img)
    if img and video:
        with open('./photo/p_' + cnt + '.png', 'wb') as fdecode:
            fdecode.write(img)
            tmp = cv2.imread('./photo/' + cnt + '.png')
            print(np.shape(tmp))
        if(cnt < 20):
            cnt += 1
            video.write(tmp)
        else:
            print("ok")
            video.release()
            video = None

        rst = make_response(jsonify({"success": 0, "msg": "you are success"}))
        rst.headers['Access-Control-Allow-Origin'] = '*'
        return rst, 201
    else:
        rst = make_response(jsonify({"error": 1001, "msg": "failed"}))
        rst.headers['Access-Control-Allow-Origin'] = '*'
        return rst, 400

    
# show photo
@app.route('/show', methods=['GET', 'POST'])
def show_txt():
    pass

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port = 5000)
