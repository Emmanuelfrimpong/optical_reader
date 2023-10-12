import cv2
from stack_images import stackImages

def read_image():
    img = cv2.imread("table.png")
    cv2.namedWindow("Camera Feed", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # img = cv2.resize(img, (700, 700))
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contorImage = img.copy()
    imageBlur= cv2.GaussianBlur(imgGrey,(5,5),1)
    imageCanny= cv2.Canny(imageBlur,50,50)
    contors, hierarchy = cv2.findContours(imageCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(contorImage,contors,-1,(0,150,5),10)
    imageArray = ([img,contorImage])
    imageStack = stackImages(imageArray,0.5)
    cv2.imshow("Camera Feed", imageStack)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
