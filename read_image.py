import cv2

from stack_images import stackImages


def get_camera_feed():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    screen_width = int(cap.get(3))
    screen_height = int(cap.get(4))
    cv2.namedWindow("Camera Feed", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.resize(frame, (screen_width, screen_height))
        contorImage = frame.copy()
        imageGrey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # imageBlur= cv2.GaussianBlur(imageGrey,(5,5),1)
        imageCanny= cv2.Canny(frame,200,200)

        contors, hierarchy = cv2.findContours(imageCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(contorImage,contors,-1,(255,0,255),7)
        imageArray = ([frame,imageCanny,contorImage])
        imageStack = stackImages(imageArray,0.5)
        cv2.imshow("Camera Feed", imageStack)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
