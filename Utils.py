import cv2
import numpy as np
import copy
import time


class FPS:
    """
    Helps in finding Frames Per Second and display on an OpenCV Image
    """

    def __init__(self):
        self.pTime = time.time()

    def update(self, img=None, pos=(20, 50), color=(255, 0, 0), scale=3, thickness=3):
        """
        Update the frame rate
        :param img: Image to display on, can be left blank if only fps value required
        :param pos: Position on the FPS on the image
        :param color: Color of the FPS Value displayed
        :param scale: Scale of the FPS Value displayed
        :param thickness: Thickness of the FPS Value displayed
        :return:
        """
        cTime = time.time()
        try:
            fps = 1 / (cTime - self.pTime)
            self.pTime = cTime
            if img is None:
                return fps
            else:
                cv2.putText(
                    img,
                    f"FPS: {int(fps)}",
                    pos,
                    cv2.FONT_HERSHEY_PLAIN,
                    scale,
                    color,
                    thickness,
                )
                return fps, img
        except:
            return 0


class PID:
    """
    PID controller
    """

    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.last_error = 0
        self.integral = 0
        self.derivative = 0
        self.output = 0
        self.min_output = 0
        self.max_output = 100

    def update(self, current_value, set_point):
        """
        returns the value after analyzinng current value
        """


def stackImages(_imgList, cols=2, scale=0.5):
    """
    Stack Images together to display in a single window
    :param _imgList: list of images to stack
    :param cols: the num of img in a row
    :param scale: bigger~1+ ans smaller~1-
    :return: Stacked Image
    """
    imgList = copy.deepcopy(_imgList)

    # make the array full by adding blank img, otherwise the openCV can't work
    totalImages = len(imgList)
    rows = (
        totalImages // cols
        if totalImages // cols * cols == totalImages
        else totalImages // cols + 1
    )
    blankImages = cols * rows - totalImages

    width = imgList[0].shape[1]
    height = imgList[0].shape[0]
    imgBlank = np.zeros((height, width, 3), np.uint8)
    imgList.extend([imgBlank] * blankImages)

    # resize the images
    for i in range(cols * rows):
        imgList[i] = cv2.resize(imgList[i], (0, 0), None, scale, scale)
        if len(imgList[i].shape) == 2:
            imgList[i] = cv2.cvtColor(imgList[i], cv2.COLOR_GRAY2BGR)

    # put the images in a board
    hor = [imgBlank] * rows
    for y in range(rows):
        line = []
        for x in range(cols):
            line.append(imgList[y * cols + x])
        hor[y] = np.hstack(line)
    ver = np.vstack(hor)
    return ver


def main():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgList = [img, img, imgGray, img, imgGray]
        imgStacked = stackImages(imgList, 2)

        cv2.imshow("stackedImg", imgStacked)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
