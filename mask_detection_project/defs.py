import cv2
import os
import numpy as np
import imutils
import serial
import time
import defs

from multiprocessing import Process, Value

def detect_mask(frame, net, output_layers):
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    layerOutputs = net.forward(output_layers)

    class_ids = []
    cofidences = []
    boxes = []

    for output in layerOutputs :
        for detection in output :
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 :
                box = detection[:4] * np.array([w,h,w,h])
                (center_x, center_y, width, height) = box.astype("int")

                x = int(center_x - width/2)
                y = int(center_y - height/2)

                boxes.append([x,y,int(width),int(height)])
                cofidences.append(float(confidence))
                class_ids.append(class_id)

    return class_ids, cofidences, boxes
    
def camera(is_mask):
    
    print("loading mask detector model")
    weightsPath = "./face_detector/yolov4-tiny-obj_best.weights"
    cfgPath = "./face_detector/yolov4-tiny-obj.cfg"
    net = cv2.dnn.readNet(weightsPath, cfgPath)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    LABELS = ["Mask", "No Mask"]
    COLORS = [(0,255,0), (0,0,255)]

    mask_frame_num = 0
    no_mask_frame_num = 0

    print("start camera()")
    video_capture = cv2.VideoCapture(0)
    while True :
        ret, frame = video_capture.read()
        frame = imutils.resize(frame, width=400)

        class_ids, cofidences, boxes = detect_mask(frame, net, output_layers)

        # 같은 물체에 대한 박스가 많은 것을 제거
        # YOLO does not apply non-maxima suppression for us, so we need to explicitly apply it.
        idxs = cv2.dnn.NMSBoxes(boxes, cofidences, 0.5, 0.3)

        if len(idxs) > 0 :
            box_list = []
            for i in idxs.flatten() :
                x, y, w, h = boxes[i]
                box_size = boxes[i][2] * boxes[i][3]
                box_list.append(box_size)
                box_list.sort(reverse=True)
                
                if boxes[i][2] * boxes[i][3] == box_list[0]:
                    detect = str(LABELS[class_ids[i]])
                    color = COLORS[class_ids[i]]
                    label = "{}: {:.2f}%".format(detect, cofidences[i] * 100)

                    # 인식되는 label의 개수 체크
                    if detect == "Mask" :
                        mask_frame_num += 1
                        no_mask_frame_num = 0

                    elif detect == "No Mask" :
                        no_mask_frame_num += 1
                        mask_frame_num = 0

                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 2초 연속으로 mask를 썼다고 인식되는 경우

        if mask_frame_num >= 2*30 :
            mask_frame_num = 0
            no_mask_frame_num = 0
            is_mask.value = 1

        # 2초 연속으로 mask를 안 썼다고 인식되는 경우
        if no_mask_frame_num >= 2*30 :
            mask_frame_num = 0
            no_mask_frame_num = 0
            is_mask.value = 0

        cv2.imshow('Video', frame)

        # q를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    is_mask.value = 2
    video_capture.release()
    cv2.destroyAllWindows()

def check_temp(arduino):
    time.sleep(3)
    c = 't'
    c = c.encode('utf-8')
    arduino.write(c)

    temp = arduino.readline()
    temp = float(temp)

    print("temp : ", temp)
    if 35.5 <= temp and temp < 37.5:
        return True

    return False

def checking(is_mask, warning_cnt) :
    arduino = serial.Serial('COM5', 9600)
    print("start checking()")
    while True :        # -1이기 때문에 무한 루프를 돈다. 마스크 감지 될 때 까지.
        if is_mask.value == -1 :
            continue
        
        elif is_mask.value == 1 : # 마스크를 썼으면 바로 체온 측정으로 넘어간다.
            is_normal = check_temp(arduino) 

            if is_normal : # 정상 체온이면!
                time.sleep(3)
                c = 'm'
                c = c.encode('utf-8')
                arduino.write(c)
                print("출입 가능합니다. 🤗")
                warning_cnt.value = 0
                arduino.close()

            elif warning_cnt.value < 2 : # 정상 체온 아니면!
                print("error : 정상 체온이 아닙니다. 🥵")
                warning_cnt.value += 1
            
            else :
                print("출입을 통제합니다. 근처 진료소를 방문하십시오. 🚫")
                warning_cnt.value = 0
                
        elif is_mask.value == 0 :
            print("error : detect no mask👿")
            warning_cnt.value = 0
            
        else :
            break

        is_mask.value = -1 # 모든 결과 후에 is_mask.value 값은 -1이 된다.

