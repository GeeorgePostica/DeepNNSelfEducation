import cv2

# Test image loading and conversion
image = cv2.imread("clouds.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Over the Clouds", image)
cv2.imshow("Over the Clouds - gray", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Test camera capture
import numpy as np

cap = cv2.VideoCapture(0)

contrast = 1.0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # print(type(gray), gray)
    # print('-'*20)
    # print(type(gray * contrast), gray * contrast)
    # print('*'*20)
    # print(type((gray * contrast).astype(int)), (gray * contrast).astype(int))
    # cv2.waitKey(0)
    # Display the resulting frame
    cv2.imshow('frame', np.minimum((gray * contrast), 255.0).astype(np.uint8))
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('+'):
        contrast += 0.1
        print('New contrast = ', contrast)
    elif key & 0xFF == ord('-'):
        contrast -= 0.1
        print('New contrast = ', contrast)
    contrast = max(0.0, min(contrast, 4.0))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()