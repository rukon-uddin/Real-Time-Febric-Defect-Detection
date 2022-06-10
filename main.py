from bokeh.plotting import ColumnDataSource, figure, output_file, curdoc
from flask import Flask, render_template, url_for, Response
from bokeh.resources import INLINE
from bokeh.embed import components
from bokeh.models import HoverTool
from execute_queries import Exec_Q
import mysql.connector
import pandas as pd
import numpy as np
import itertools
import time
import os
from turtle import st
import cv2
from cv2 import dnn
from scipy.spatial import distance as dist
from datetime import  datetime


camera = cv2.VideoCapture(0+cv2.CAP_DSHOW)
app = Flask(__name__)

def get_area(bbox):
    by1, bx1, by2, bx2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
    return (bx2 - bx1) * (by2 - by1)


def get_overlap(bbox, tbox):
    by1, bx1, by2, bx2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
    ty1, tx1, ty2, tx2 = tbox[0], tbox[1], tbox[0] + tbox[2], tbox[1] + tbox[3]
    overlapx1 = max(bx1, tx1)
    overlapy1 = max(by1, ty1)
    overlapx2 = min(bx2, tx2)
    overlapy2 = min(by2, ty2)
    overlap_area = (overlapx2 - overlapx1) * (overlapy2 - overlapy1)
    bbox_area = get_area(bbox)
    tbox_area = get_area(tbox)
    smaller_a = tbox_area if tbox_area < bbox_area else bbox_area
    epsilon = 1e-5
    return (overlap_area) / (smaller_a + epsilon)


def findObject(output, img, thresh, threshold=0.3, overlap_threshold=0.5):  # t  = 0.6, ot = 0.6
    # d=dict() # Dictionary used to count the number of certain objects in the current frame

    global classes
    height, width, channel = img.shape
    bbox = []
    valid_bbox = []
    classId_list = []
    valid_class_list = []
    confidence_list = []
    valid_confidence_list = []

    for out in output:  # Go through each output
        for det in out:  # check confidence value fro each class
            conf_score = det[5:]
            classId = np.argmax(conf_score)  # store the max confidence value class
            confidence = conf_score[classId]

            if confidence > threshold:
                # det[0],det[1],det[2],det[3] are returned as percentages so we need to convert them by multiplying image width
                # and height
                w, h = int(det[2] * width), int(det[3] * height)
                startX, startY = int((det[0] * width) - w / 2), int((det[1] * height) - h / 2)
                endX = int(startX + w)
                endY = int(startY + h)
                bbox.append([startY, startX, endY, endX])
                classId_list.append(classId)
                confidence_list.append(float(confidence))

    index = cv2.dnn.NMSBoxes(bbox, confidence_list, threshold, overlap_threshold)  # applying non max suppresion
    # print(index)
    for i in index:
        i = i
        (sY, sX, eY, eX) = bbox[i]
        mX = int((sX + eX) / 2)
        mY = int((sY + eY) / 2)
        # centY=int((sY+eY)/2)

        # if mX <= int(0.35 * width) and mY < int(0.8*height):
        # if mX <= 95:
        #     continue
        # elif mY<int(0.5*height):
        # continue
        valid_bbox.append(bbox[i])
        valid_class_list.append(classes[classId_list[i]])
        valid_confidence_list.append(confidence_list[i])

    return valid_bbox, valid_class_list, valid_confidence_list


##################################################################


###################################################################

print(os.getcwd())
os.chdir(r"D:\Users\rukon\Desktop\capstone project\febric defect\models\yolo v3 tiny\tracker")

dirname = os.getcwd()
dirname = os.path.dirname(__file__)
model_weights = os.path.join(dirname, "mix_w/yolov3-tiny_fdd_4000.weights")
model_config = os.path.join(dirname, "mix_w/yolov3-tiny_fdd.cfg")

names_file = os.path.join(dirname, 'classes-tiny.txt')

with open(names_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n');

net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


def generate_frames():
    db = Exec_Q()
    W = 480
    H = 800
    exp = 0

    thresh = int(W * 0.36)
    sec_dic = {}

    sec_dic["g"] = 0
    sec_dic["b"] = 0
    sec_dic["o"] = 0

    coord = {"coords":[]}

    last_time = time.time()
    current_y = 0
    rukk = 0
    camera = cv2.VideoCapture(r"test2.mp4", cv2.CAP_FFMPEG)
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Disconnected")
            camera.release()
            break
        frame = cv2.resize(frame, (W, H))
        mono_defect = frame.copy()

        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (W, H), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layerNames = net.getLayerNames()
        outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
        output = net.forward(outputNames)
        rects, names, confidence = findObject(output, frame, thresh)

        cv2.line(frame, (0, 300), (W, 300), (0, 0, 0), 2)
        
        for ((startY, startX, endY, endX), n, c) in itertools.zip_longest(rects, names, confidence):
            (sY, sX, eY, eX) = (startY, startX, endY, endX)
            mX = int((sX + eX) / 2)
            mY = int((sY + eY) / 2)
            
            y_counter = time.time() - last_time
            y_counter = int(y_counter % 60)
            if 297 <= mY <= 300:
                current_y = y_counter + current_y + exp
                coord['coords'].append([mX, current_y])
                cv2.rectangle(mono_defect, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.circle(mono_defect, (mX, mY), 2, (0, 0, 255), -2)
                cv2.imwrite(r"D:\Users\rukon\Desktop\capstone project\models\Full_system\static\{}.jpg".format(rukk), mono_defect) 
                INSERT_QUERY = "INSERT INTO defects (x, y, img) VALUES (%s, %s, %s)"
                row = (str(mX), str(current_y), str("{}.jpg".format(rukk)))
                db.insert_row(INSERT_FORMULA=INSERT_QUERY, rows=row)
                rukk+=1         

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.circle(frame, (mX, mY), 2, (0, 0, 255), -2)
 

            c = round(c, 2)
            cv2.putText(frame, str(c), (startX, startY - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            cv2.putText(frame, str(n), (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        # print(len(coord['coords']))
        # cv2.imshow('counter',frame)
        # cv2.waitKey(1)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        exp+=1
        yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/graph")
def graph():
    mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    passwd = "root",
    database = "defected_img")
    mycursor = mydb.cursor()
    SELECT_ALL = "SELECT x FROM defects"
    mycursor.execute(SELECT_ALL)
    result = mycursor.fetchall()
    X = []
    for r in result:
        t = int(r[0])
        X.append(t)

    SELECT_ALL = "SELECT y FROM defects"
    mycursor.execute(SELECT_ALL)
    result = mycursor.fetchall()
    Y = []
    for r in result:
        t = int(r[0])
        Y.append(t)


    SELECT_ALL = "SELECT img FROM defects"
    mycursor.execute(SELECT_ALL)
    result = mycursor.fetchall()
    img = []
    for r in result:    
        t = r[0]
        t = '/static/{}'.format(t)
        img.append(t)
    print(X)

    source = ColumnDataSource()
    source.data =dict(x=X, y=Y, imgs=img)
    TOOLTIPS = """
    <div>
        <div>
            <img
                src="@imgs" height="250" alt="@imgs" width="200"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
        </div>
    </div>
    """

    fig = figure(plot_width=450, plot_height=800, tooltips = TOOLTIPS, title="Mouse over the dots")
    fig.circle('x', 'y', size=20, source=source)
    script, div = components(fig)

    output_file("toolbar.html")
    return render_template(
        'graph.html',
        plot_script=script,
        plot_div=div,
        js_resources=INLINE.render_js(),
        css_resources=INLINE.render_css(),
        )
    


@app.route("/", methods=['GET', 'POST'])
def hello_world():
    return render_template('index.html',)


@app.route("/video")
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
