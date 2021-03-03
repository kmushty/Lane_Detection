import cv2
import numpy as np



cap = cv2.VideoCapture('Night Drive - 2689.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

while (cap.isOpened()):
    ret, frame = cap.read()

    Gauss = cv2.GaussianBlur(frame, (5, 5), 0)

    hsv = cv2.cvtColor(Gauss, cv2.COLOR_BGR2HSV)
    hsv_v = hsv[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(20, 20))  # increased values may cause noise
    cl1 = clahe.apply(hsv_v)
    gamma = 1.4


    def adjust_gamma(image, gamma=1.0):

        Look_Up_table = np.array([((i / 255.0) ** (1/gamma)) * 255 for i in np.arange(0, 256)])
        return cv2.LUT(image.astype(np.uint8), Look_Up_table.astype(np.uint8))
    cl1 = adjust_gamma(cl1)
    hsv[:, :, 2] = cl1
    improved_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('improved_image', improved_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# releasing the video feed
cap.release()
out.release()
cv2.destroyAllWindows()