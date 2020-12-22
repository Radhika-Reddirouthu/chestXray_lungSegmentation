#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Importing libraries
import os
import numpy as np
import classifier as clf
import local_config as lc
from functools import wraps, update_wrapper
from datetime import datetime
from flask import Flask, request, redirect, url_for, render_template,make_response
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from flask import session

# Setting up environment
if not os.path.isdir(lc.OUTPUT_DIR):
    print('Creating static folder..')
    os.mkdir(lc.OUTPUT_DIR)

app = Flask(__name__)
Bootstrap(app)
app.config['UPLOAD_FOLDER'] = lc.OUTPUT_DIR
app.secret_key = 'super secret key'
#app.config['SESSION_TYPE'] = 'filesystem'
#session.init_app(app)
#app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['Cache_Control']='public,max-age=0'
        return response   
    return update_wrapper(no_cache, view)


@app.route('/', methods=['GET', 'POST'])
@nocache
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # Check if no file was submitted to the HTML form
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and clf.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            clf.lung_segmentation(filename)
            user_input=url_for('static',filename = filename)
            filename=secure_filename("my.png")
            path_to_image = url_for('static', filename = filename)
            #path_to_image="../my.png"
            result = {
                'user_input': user_input,
                'path_to_image': path_to_image,
                'size': lc.SIZE
            }
            session.clear()

            return render_template('show.html', result=result)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

    