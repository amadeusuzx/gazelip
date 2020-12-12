
 
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
    counter = 0
    while True:
        left = global_gaze_data["left_gaze_point_on_display_area"]
        right = global_gaze_data["right_gaze_point_on_display_area"]
        if not (math.isnan(left[0]) or math.isnan(right[0])):
            curr_gaze_point = np.array([(left[0]+right[0])/2,(left[1]+right[1])/2])
            dis = np.linalg.norm((curr_gaze_point - opti_gaze_point))
            if dis > 0.1:
                counter=0
                opti_gaze_point = opti_gaze_point * 0.3 + curr_gaze_point * 0.7
            elif dis > 0.05:
                counter=0
                opti_gaze_point = opti_gaze_point * 0.7 + curr_gaze_point * 0.3
            else:
                counter+=1
                if counter > 15:
                    pyautogui.click()
                    counter = 0
                opti_gaze_point = opti_gaze_point * 0.7 + curr_gaze_point * 0.3
        
                
            pyautogui.moveTo(opti_gaze_point[0]*1920*2,opti_gaze_point[1]*1080*2)

if __name__ == "__main__":

    pyautogui.FAILSAFE = False
    global_gaze_data = None
    
    for eyeTracker in tr.find_all_eyetrackers():

        print("find eyetracker")
        gaze_data(eyeTracker)
