import darknet
import pika
import time
import json
import os
import cv2
from PIL import Image
from io import StringIO, BytesIO
import numpy as np


netMain=None
metaMain=None
altNames=None

def send_to_rabbit(load_json):
    connection=pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel=connection.channel()
    channel.queue_declare(queue='klass',durable=True)
    channel.basic_publish(exchange='',routing_keys='klass',body=load_json)
    connection.close()


def convertBack(x,y,w,h):
    xmin=int(round(x-(w/2)))
    xmax=int(round(x+(w/2)))
    ymin=int(round(y-(h/2)))
    ymax=int(round(y+(h/2)))
    return xmin,ymin,xmax,ymax


def cvDrawBoxes(detection,img):
    for detection in detections:
        x,y,w,h=detection[2][0],detection[2][1],detection[2][2],detection[2][3]
        xmin,ymin,xmax,ymax=convertBack(float(x),float(y),float(w),float(h))
        pt1=(xmin,ymin)
        pt2=(xmax,ymax)
        cv2.rectangle(img,pt1,pt2,(0,255,0),1)
        cv2.putText(img,detection[0].decode()+"["+ str(round(detection[1]*100,2))+"]",
                    (pt1[0],pt1[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,[0,255,0],2)


def readb64(base64_string):
    im=Image.open(BytesIO(base64.b64decode(base64_string)))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def detectionToJson(detections):
    #TODO
    pass


def LoadWeight():
    global metaMain, netMain, altNames
    configPath=""
    weightPath=""
    metaPath=""

    if not os.path.exists(configPath):
        raise ValueError("Invalid config path "+ os.path.abspath(configPath))
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path "+ os.path.abspath(weightPath))
    if not os.path.exists(metaPath):
        raise ValueError("Invalid meta path "+ os.path.abspath(metaPath))

    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode("ascii"),weightPath.encode("ascii"),0,1)
    if metaMain is None:
        metaMain= darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents=metaFH.read()
                import re
                match = re.search("names *= *(-*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result=match.group(1)
                else:
                    result=None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            nameList=namesFH.read().strip().split("\n")
                            altNames=[x.strip() for x in nameList]

                except TypeError:
                    pass ##TODO
        except Exception:
            pass ##TODO


def detect_image(image,thresh):
    prev_time=time.time()

    #Create image
    darknet_image=darknet.make_image(darknet.network_width(netMain),
                                     darknet.network_height(netMain),3)

    img=image

    print("DEBUG - start yolo loop")


    frabe_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_resized=cv2.resize(frame_rgb, (darknet.network_width(netMain),
                                         darknet.network_height(netMain)),
                                        interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
    detection=darknet.detect_image(netMain,metaMain,darknet_image,thresh=thresh)

    return detection




def callback(ch,method,properties,body):
    start_time=time.time()
    jsonload=json.loads(body)
    #DEBUG
    cvimg=readb64(jsonload["photo"])
    cv2.imshow("img",detect_image(cvimg))
    cv2.waitKey(0)

    cvimag=cv2.imread(jsonload["path"])##TODO


    #MAKE DETECTION
    #TODO

#TODO check the json get from OpenAlpr

##TODO create function for rabbitmq get json

#TODO create fuctnion that put json modification and send to rabbitmq

#TODO get image from the link with request