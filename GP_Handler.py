from inputs import get_gamepad
import time
import threading

# while True:
#     events = get_gamepad()
#     for event in events:
#         print(event.code, event.state)

class GPReader():
    def __init__(self):
        self._BTN_WEST = 0
        self._BTN_SOUTH = 0
        self._BTN_EAST = 0
        self._BTN_NORTH = 0
        self._ABS_HAT0X = 0
        self._ABS_GAS = 0
        self._ABS_BRAKE = 0
        self._BTN_TR = 0
        self._BTN_TL = 0
        self._BTN_SELECT = 0
        self._BTN_START = 0
        self._runTime = time.time()

        self._GPThread = threading.Thread(target=self.Run, args=())
        self._GPThread.start()

    def Reset(self):
        self._BTN_WEST = 0
        self._BTN_SOUTH = 0
        self._BTN_EAST = 0
        self._BTN_NORTH = 0
        self._ABS_HAT0X = 0
        self._ABS_GAS = 0
        self._ABS_BRAKE = 0
        self._BTN_TR = 0
        self._BTN_SELECT = 0
        self._BTN_START = 0
        self._BTN_TL = 0

    def Stop(self):
        self._BTN_START = 1
        self._GPThread.join()

    def Run(self):
        try:
            while not self._BTN_START:
                if time.time() - self._runTime > 5:
                    self.Reset()
                    self._runTime = time.time()

                events = get_gamepad()
                for event in events:
                    if event.code == 'BTN_WEST':
                        self._BTN_WEST = event.state
                    elif event.code == 'BTN_SOUTH':
                        self._BTN_SOUTH = event.state
                    elif event.code == 'BTN_EAST':
                        self._BTN_EAST = event.state
                    elif event.code == 'BTN_NORTH':
                        self._BTN_NORTH = event.state
                    elif event.code == 'ABS_HAT0X':
                        self._ABS_HAT0X = event.state
                    elif event.code == 'ABS_GAS':
                        self._ABS_GAS = event.state
                    elif event.code == 'ABS_BRAKE':
                        self._ABS_BRAKE = event.state
                    elif event.code == 'BTN_TR':
                        self._BTN_TR = event.state
                    elif event.code == 'BTN_TL':
                        self._BTN_TL = event.state
                    elif event.code == 'BTN_SELECT':
                        self._BTN_SELECT = event.state
                    elif event.code == 'BTN_START':
                        self._BTN_START = event.state
        except:
            self._BTN_START = 1

        print("STOPPPPPPPPPPPPPPPPPPP")
