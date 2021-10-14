import cv2

cam = cv2.VideoCapture(0)

w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = max(cam.get(cv2.CAP_PROP_FPS) % 100, 0)

frames = max(int(cam.get(cv2.CAP_PROP_FRAME_COUNT)), 0)

frame_bool, img = cam.read()

print('Is camera connected? ', cam.isOpened())

print('Camera FPS: ', fps)

print('Camera frame: ', frames)

print('Has frame been grabbed? ', frame_bool)

print('img size: ', img.shape)