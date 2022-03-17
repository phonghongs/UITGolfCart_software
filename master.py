import asyncio, socket
import re
import datetime
from tkinter.tix import Tree
from Libs.utils import SimpleKalmanFilter, load_coefficients
from albumentations.pytorch import ToTensorV2
import cv2
import threading
import struct
import math
import numpy as np
import albumentations as A

global raw_image, done, blank_image
raw_image = []
blank_image = np.zeros((256,512,3), np.uint8)
done = False

ip = "0.0.0.0"
port = 15555
MAX_DGRAM = 2**16
MAX_IMAGE_DGRAM = MAX_DGRAM - 64 # extract 64 bytes in case UDP frame overflown

mtx2, dist2 = load_coefficients('CalibCamera_96/calibration_chessboard_96deg.yml')

test_transform = A.Compose(
    [
        A.Resize(height=480, width=600),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            # max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

def camReader():
    global raw_image, done
    cap = cv2.VideoCapture('test.mp4')
    while (not done) and (cap.isOpened()):
        ret, img = cap.read()
        if ret:
            image_resized = cv2.resize(img[50:], (512, 256))
            raw_image = cv2.undistort(image_resized, mtx2, dist2, None, None)
        else:
            raw_image = blank_image

        cv2.waitKey(10)

def dataJetson1(image):
    return image

def dataJetson2(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return image

class FrameSegment(object):
    """
    Object to break down image frame segment
    if the size of image exceed maximum datagram size 
    """
    MAX_DGRAM = 2**16
    MAX_IMAGE_DGRAM = MAX_DGRAM - 64 # extract 64 bytes in case UDP frame overflown

    def __init__(self, loop, client):
        self.loop = loop
        self.client = client

    async def udp_frame(self, img):
        """
        Compress image and Break down
        into data segments
        """
        compress_img = cv2.imencode('.jpg', img)[1]
        dat = compress_img.tostring()
        print(type(compress_img))
        size = len(dat)
        count = math.ceil(size/(MAX_IMAGE_DGRAM))
        array_pos_start = 0
        while count:
            array_pos_end = min(size, array_pos_start + MAX_IMAGE_DGRAM)
            await loop.sock_sendall(self.client, (struct.pack("B", count) +
                dat[array_pos_start:array_pos_end])
                )
            array_pos_start = array_pos_end
            count -= 1


async def handle_client(client):
    global raw_image

    loop = asyncio.get_event_loop()
    request = None

    fs = FrameSegment(loop, client)

    while request != 'quit':
        request = (await loop.sock_recv(client, 255)).decode('utf8')
        image = np.copy(raw_image)
        # print(request)
        if (request == 'JETSON1'):
            await fs.udp_frame(raw_image)
        elif (request == 'JETSON2'):
            await fs.udp_frame(raw_image)        
        elif request != 'quit':
            await fs.udp_frame(image)

    client.close()


async def run_server():
    global done
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((ip, port))
    server.listen(8)
    server.setblocking(False)
    loop = asyncio.get_event_loop()
  
    camreader = threading.Thread(target=camReader)
    camreader.start()

    while not done:
        client, _ = await loop.sock_accept(server)
        loop.create_task(handle_client(client))

loop = asyncio.get_event_loop()
loop.run_until_complete(run_server())
