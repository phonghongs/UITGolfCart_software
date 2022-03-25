import cv2
import numpy as np
import time
import keyboard 
import threading
import os

from simple_pid import PID
from net import Net
from src.util import adjust_fits, get_steer_angle, calcul_speed
from src.parameters import Parameters
from utils.connect import Connection
from utils.controller import Controller
from inputs import get_gamepad
from scipy import stats
from GP_Handler import GPReader
import matplotlib.pyplot as plt
import math

global count, MODE, DONE, DEBUG, CENTER_LINE, baseAngle
count = 0
baseAngle = 0
DONE = False
MODE = 1
DEBUG = True

MAX_SPEED = 80
MAX_ANGLE = 25

CENTER_LINE = 220


def Thread_read_key_board():
    global MODE, DONE, GP_OK
    
    GP_OK = True

    gamePad = GPReader()

    while cap.isOpened() and not gamePad._BTN_START:
        if MODE == 1:
            print("Car controller: ", car_controller.get_info())
            if gamePad._BTN_WEST:
                car_controller.increase_mode()

            elif gamePad._BTN_SOUTH:
                car_controller.decrease_mode()

            elif gamePad._BTN_EAST:
                car_controller.increase_speed(2)

            elif gamePad._BTN_NORTH:   
                car_controller.decrease_speed(2)
            
            elif gamePad._ABS_HAT0X == 1:
                car_controller.turn_right(1)

            elif gamePad._ABS_HAT0X == -1 :
                car_controller.turn_left(1)

            if gamePad._ABS_GAS:
                car_controller.go_straight()  
                print("Di thang")

            elif gamePad._BTN_TR:
                car_controller.go_reverse()  

            elif gamePad._ABS_BRAKE == 255:
                car_controller.brake()  
            else:
                car_controller.nop()

        if gamePad._BTN_TL == 1 and MODE == 0:
            MODE = 1
            car_controller.brake()  
        
        if gamePad._ABS_BRAKE == 255:
            car_controller.brake() 

        if gamePad._BTN_SELECT:
            print("HRERERERE")
            if MODE == 1:
                print("Auto")
                MODE = 0     
            else:       
                print("Manual")   
                MODE = 1
                car_controller.brake()  

        if gamePad._BTN_START:
            gamePad.Stop()
            DONE = True
        time.sleep(0.1)

    gamePad.Stop()
    DONE = True
    GP_OK = False



def GetRealAngle(angle):
    global baseAngle
    if abs(baseAngle - angle) < 1:
        return baseAngle
    if baseAngle < angle:
        baseAngle = np.clip(baseAngle +0.5, -25, 25)
    elif baseAngle > angle:
        baseAngle = np.clip(baseAngle - 0.5, -25, 25)

    return baseAngle


if __name__ == "__main__":

    # connector = Connection("/dev/ttyTHS2", 115200, False)
    connector = Connection("/dev/ttyTHS2", 115200, True)
    connector.Car_ChangeMaxSpeed(MAX_SPEED)
    car_controller = Controller(connector)

    main_ThreadKeyBoard = threading.Thread(target= Thread_read_key_board)
    main_ThreadKeyBoard.start()
    time.sleep(5)

    main_ThreadKeyBoard.join()

    print("END THE PROGRAM")
