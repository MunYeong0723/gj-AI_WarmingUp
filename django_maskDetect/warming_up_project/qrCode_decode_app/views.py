from django.shortcuts import render, redirect
from django.http.response import StreamingHttpResponse

import cv2
import pyzbar.pyzbar as pyzbar


# Create your views here.
def qrDecode(request):
    return render(request, 'qrDecode.html')

def viewCamera(request):
    video_capture = cv2.VideoCapture(0)
    while True :
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        decoded = pyzbar.decode(gray)

        data = ""
        if len(decoded) == 1 :
            barcode_data = decoded[0].data.decode("utf-8")

            if len(barcode_data) > 0 :
                data = barcode_data

        if len(data) > 0 :
            video_capture.release()
            cv2.destroyAllWindows()

            # TODO : 개인정보를 저장해두기 (db or something else)
            print("😃")
            print(data)

            return redirect('maskDetect')

        cv2.imshow('QR code', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return redirect('home')
