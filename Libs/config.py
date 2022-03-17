import argparse

serialPort = 'COM8'
serialBaudrate = 115200
weight = "./Laneline/checkpoint/15_epoch.pkl"

def args_setting():
    # Training settings
    parser = argparse.ArgumentParser(description='GOLFCAR_CONFIG')
    parser.add_argument('--serialtest', type=bool, default=True, metavar='S',
                        help='True/False : enable/disable PC test mode')
    parser.add_argument('--maxspeed', type=int, default=70, metavar='MS',
                        help='set golf cart max speed')
    args = parser.parse_args()
    return args
