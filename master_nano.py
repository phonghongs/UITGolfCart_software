import asyncio, socket
import re
import datetime
from tkinter.tix import Tree
from Libs.laneprocess import CenterCalv2
from Libs.utils import SimpleKalmanFilter, load_coefficients
from albumentations.pytorch import ToTensorV2
import cv2
import threading
import struct
import math
import numpy as np
import albumentations as A
import ast

global raw_image, done
raw_image = []
done = False

ip = "0.0.0.0"
port = 15555
MAX_DGRAM = 2**16
MAX_IMAGE_DGRAM = MAX_DGRAM - 64 # extract 64 bytes in case UDP frame overflown

mtx2, dist2 = load_coefficients('CalibCamera_96/calibration_chessboard_96deg.yml')

def dataJetson1(dataPoint=''):
    print(type(dataPoint))
    xStr, yStr = dataPoint.split(';')
    x = ast.literal_eval(xStr)
    y = ast.literal_eval(yStr)

    blank_image = np.zeros((256, 512,3), np.uint8)
    point, image_points_center = CenterCalv2(x, y, blank_image, 225)
    cv2.imshow('RESULT', image_points_center)
    cv2.waitKey(1)

def dataJetson2(dataSeg=''):
    contData = ast.literal_eval(dataSeg)
    blank_image = np.zeros((256, 512,3), np.uint8)
    for cnt in contData:
        cv2.circle(blank_image, (cnt[0], cnt[1]), 1, 255, 2)
    
    cv2.imshow("Segment", blank_image)
    cv2.waitKey(1)


async def handle_client(client):
    global raw_image

    loop = asyncio.get_event_loop()
    request = None
    dat = b''
    done = False

    while request != 'quit':
        request = (await loop.sock_recv(client, 2**16 - 64)).decode('utf8')
        
        node, data = request.split(':')
        if node == 'TX2_1':
            dataJetson1(data)
        elif node == 'TX2_2':
            dataJetson2(data)

        await loop.sock_sendall(
            client,
            'ok'.encode('utf8')
        )

    client.close()


async def run_server():
    global done
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((ip, port))
    server.listen(8)
    server.setblocking(False)
    loop = asyncio.get_event_loop()

    while not done:
        client, _ = await loop.sock_accept(server)
        loop.create_task(handle_client(client))


loop = asyncio.get_event_loop()
loop.run_until_complete(run_server())
