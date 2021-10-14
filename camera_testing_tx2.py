import cv2

gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1920, height=(int)1080, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=2 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(1920, 1080)


cam = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = max(cam.get(cv2.CAP_PROP_FPS) % 100, 0)

frames = max(int(cam.get(cv2.CAP_PROP_FRAME_COUNT)), 0)

#cam.open(0)
frame_bool, img = cam.read()

print('Is camera connected? ', cam.isOpened())

print('Camera FPS: ', fps)

print('Camera frame: ', frames)

print('Has frame been grabbed? ', frame_bool)

print('img size: ', img.shape)
