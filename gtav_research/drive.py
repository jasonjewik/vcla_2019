"""
Car driving module.
"""

import time

# gamepad axes limits and gamepad module
from gamepad import AXIS_MIN, AXIS_MAX, TRIGGER_MAX, XInputDevice, BTN_ON, BTN_OFF

def countdown(duration=5):
    for i in list(range(duration))[::-1]:
        print(i+1)
        time.sleep(1)
        
def press_and_release(controller, button):
    controller.SetBtn(button, BTN_ON)
    time.sleep(1)
    controller.SetBtn(button, BTN_OFF)
    
def enter_vehicle(controller):
    countdown(7)
    press_and_release(controller, "Y")
    
def drive_vehicle(controller, duration):
    countdown(3)
    controller.SetTrigger('R', TRIGGER_MAX)
    time.sleep(duration)
    controller.SetTrigger('R', 0)
    
def main():
    controller = XInputDevice(1)
    controller.PlugIn()
    enter_vehicle(controller)
    drive_vehicle(controller, 5)
    controller.UnPlug()
    
if __name__ == '__main__':
    main()