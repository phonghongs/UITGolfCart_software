
from __future__ import division
from pickle import TRUE
import cv2
import numpy as np
import socket
import struct
from Libs.config import args_setting
from Libs import config
from Libs.laneprocess import CenterCalv2
from Laneline.net import Net

MAX_DGRAM = 2**16
CENTER_LINE = 220
CLIENT_ID = "JETSON1"

def dump_buffer(s):
    """ Emptying buffer frame """
    while True:
        seg, addr = s.recvfrom(MAX_DGRAM)
        print(seg[0])
        if struct.unpack("B", seg[0:1])[0] == 1:
            print("finish emptying buffer")
            break


def main():
    """ Getting image udp frame &
    concate before decode and output image """

    # Set up socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('localhost', 15555))
    dat = b''
    done = False
    # dump_buffer(s)

    net = Net()
    net.load_model(config.weight)

    print("OK")
    while not done:
        streamStart = True
        s.send(CLIENT_ID.encode('utf8'))
        while streamStart:
            seg = s.recv(MAX_DGRAM)
            if struct.unpack("B", seg[0:1])[0] > 1:
                dat += seg[1:]
            else:
                dat += seg[1:]
                img = cv2.imdecode(np.fromstring(dat, dtype=np.uint8), 1)

                y , x = net.predict(img, warp = False)
                point, image_points_center = CenterCalv2(x, y, img, CENTER_LINE)

                cv2.imshow("image_points_center", image_points_center)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    done = True

                dat = b''
                streamStart = False

    s.send("quit".encode('utf8'))

    cv2.destroyAllWindows()
    s.close()

if __name__ == "__main__":
    main()
