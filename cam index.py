import cv2

def find_camera(max_index=10):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index {i}")
            cap.release()
        else:
            print(f"no camera at index {i}")

find_camera()