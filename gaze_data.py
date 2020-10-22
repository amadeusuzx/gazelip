
 
 # <BeginExample>
import time
import tobii_research as tr
import math
import pyautogui
import numpy as np


def gaze_data_callback(gaze_data):
    global global_gaze_data
    global_gaze_data = gaze_data
 

def gaze_data(eyetracker):
    global global_gaze_data
    opti_gaze_point = np.array([0,0])
    print("Subscribing to gaze data for eye tracker with serial number {0}.".format(eyetracker.serial_number))
    eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)

    # Wait while some gaze data is collected.
    time.sleep(2)
    while True:
        left = global_gaze_data["left_gaze_point_on_display_area"]
        right = global_gaze_data["right_gaze_point_on_display_area"]
        if not (math.isnan(left[0]) or math.isnan(right[0])):
            curr_gaze_point = np.array([(left[0]+right[0])/2,(left[1]+right[1])/2])
            dis = np.linalg.norm((curr_gaze_point - opti_gaze_point))
            if dis > 0.1:
                opti_gaze_point = curr_gaze_point
            elif dis > 0.05:
                opti_gaze_point = opti_gaze_point * 0.7 + curr_gaze_point * 0.3
            else:
                opti_gaze_point = opti_gaze_point * 0.9 + curr_gaze_point * 0.1
            pyautogui.moveTo(opti_gaze_point[0]*1920*2,opti_gaze_point[1]*1080*2)
    eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
    print("Unsubscribed from gaze data.")

    print("Last received gaze package:")
    print(global_gaze_data)
if __name__ == "__main__":

    pyautogui.FAILSAFE = False
    global_gaze_data = None
    for eyeTracker in tr.find_all_eyetrackers():
        gaze_data(eyeTracker)
    
