"""
Pose Module
modified accoring to the needs to control tello
"""

import cv2
import mediapipe as mp
from math import degrees, acos, hypot, atan

from Utils import stackImages, FPS


class PoseModule:
    """
    Estimates Pose points of a human body using the mediapipe library.
    """

    def __init__(
        self,
        body=True,
        face=True,
        hands=True,
        static=False,
        complx=1,
        smooth=True,
        detectionCon=0.5,
        trackCon=0.5,
    ):
        """
        :param body: Flag to enable the body detection.
        :param face: Flag to enable the face detection.
        :param hands: Flag to enable the hands detection.
        :param static: In static mode, detection is done on each image: slower
        :param complx: Upper body only flag
        :param smooth: Smoothness Flag
        :param detectionCon: Minimum Detection Confidence Threshold
        :param trackCon: Minimum Tracking Confidence Threshold
        """

        self.body = body
        self.face = face
        self.hands = hands

        self.results = None

        self.mpDraw = mp.solutions.drawing_utils

        self.mpPose = mp.solutions.holistic
        self.pose = self.mpPose.Holistic(static, complx, smooth, detectionCon, trackCon)

    def processPose(self, image, draw=True):
        """process the image to find the pose

        Args:
            img (image): image to process
            draw (bool, optional): flag to draw the output. Defaults to True.
        """
        img = image.copy()
        self.h, self.w, _ = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if draw and self.results is not None:
            if self.body and self.results.pose_landmarks:
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS
                )
            if self.hands and self.results.left_hand_landmarks:
                self.mpDraw.draw_landmarks(
                    img, self.results.left_hand_landmarks, self.mpPose.HAND_CONNECTIONS
                )
            if self.hands and self.results.right_hand_landmarks:
                self.mpDraw.draw_landmarks(
                    img, self.results.right_hand_landmarks, self.mpPose.HAND_CONNECTIONS
                )
            if self.face and self.results.face_landmarks:
                self.mpDraw.draw_landmarks(
                    img, self.results.face_landmarks, self.mpPose.FACE_CONNECTIONS
                )

        return img

    def processBBox(self, image, draw=True, trim=True, crop=False):
        """find postion of the human in the image

        Args:
            img (image): image to find position in
            draw (bool, optional): flag to draw the output. Defaults to True.
            crop (bool, optional): flag to return cropped output. Defaults to False.

        Returns:
            image: image with or without drawings
        """
        lmx = []
        lmy = []

        img = image.copy()

        if self.results is None:
            return img

        if self.results.pose_landmarks:

            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                lmx.append(int(lm.x * self.w))
                lmy.append(int(lm.y * self.h))

            # Bounding Box

            dx = 20 + abs(lmx[12] - lmx[11]) // 2
            dy = 20 + abs(lmy[23] - lmy[11]) // 2

            xl = min(lmx) - dx
            xr = max(lmx) + dx

            yu = min(lmy) - dy
            yd = max(lmy) + dy

            bbox = (xl, yu, abs(xr - xl), abs(yd - yu))
            cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + bbox[3] // 2

            # self.bboxInfo = {"bbox": bbox, "center": (cx, cy)}

            if draw:
                cv2.rectangle(img, bbox, (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            yu = max(0, yu)
            yd = min(img.shape[0], yd)
            xl = max(0, xl)
            xr = min(img.shape[1], xr)

            if crop:
                img = img[yu:yd, xl:xr]
            elif trim:
                img[0:yu, :] = 0
                img[yd:, :] = 0
                img[:, 0:xl] = 0
                img[:, xr:] = 0

        return img

    def __getLandmarks(self):
        """return the landmarks of the human

        Returns:
            list: list of pose, rhand, lhand, face landmarks
        """

        self.poseLmList = []
        self.poseWLmList = []
        self.rhandLmList = []
        self.lhandLmList = []
        self.faceLmList = []

        if self.results.pose_world_landmarks:
            for id, lm in enumerate(self.results.pose_world_landmarks.landmark):
                self.poseWLmList.append([id, lm.x, lm.y, lm.z])

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy, cz = int(lm.x * self.w), int(lm.y * self.h), int(lm.z * self.h)
                self.poseLmList.append([id, cx, cy, cz])

        if self.results.left_hand_landmarks:
            for id, lm in enumerate(self.results.left_hand_landmarks.landmark):
                cx, cy = int(lm.x * self.w), int(lm.y * self.h)
                self.lhandLmList.append([id, cx, cy])

        if self.results.right_hand_landmarks:
            for id, lm in enumerate(self.results.right_hand_landmarks.landmark):
                cx, cy = int(lm.x * self.w), int(lm.y * self.h)
                self.rhandLmList.append([id, cx, cy])

        if self.results.face_landmarks:
            for id, lm in enumerate(self.results.face_landmarks.landmark):
                cx, cy = int(lm.x * self.w), int(lm.y * self.h)
                self.faceLmList.append([id, cx, cy])

        return [
            self.poseWLmList,
            self.poseLmList,
            self.rhandLmList,
            self.lhandLmList,
            self.faceLmList,
        ]

    def __fingersUp(self, lmList):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        fingers = []
        tipIds = [4, 8, 12, 16, 20]
        # 4 Fingers
        try:
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        except:
            return []
        finally:
            return fingers

    def getInfo(self):
        """return the pose"""
        pose = None
        rockPose = [1, 0, 0, 1]
        openPose = [1, 1, 1, 1]
        closePose = [0, 0, 0, 0]

        tilt = None
        target = None
        distance = None

        if self.results is None:
            return pose, target, distance, tilt

        self.__getLandmarks()

        if self.poseLmList:

            x1, y1, z1 = (
                self.poseLmList[11][1],
                self.poseLmList[11][2],
                self.poseLmList[11][3],
            )
            x2, y2, z2 = (
                self.poseLmList[12][1],
                self.poseLmList[12][2],
                self.poseLmList[12][3],
            )
            distance = int(hypot(x2 - x1, y2 - y1, z2 - z1))
            target = (
                self.poseLmList[0][1],
                self.poseLmList[0][2],
            )
            try:
                tilt = int(degrees(atan((z2 - z1) / (x2 - x1))))
            except:
                tilt = 90

            rUp = False
            lUp = False
            # left hand up
            if self.poseLmList[15][2] < self.poseLmList[11][2]:
                lUp = True
                lFin = self.__fingersUp(self.lhandLmList)
            # right hand up
            if self.poseLmList[16][2] < self.poseLmList[12][2]:
                rUp = True
                rFin = self.__fingersUp(self.rhandLmList)

            if lUp and rUp:  # "photo"
                pose = "BOTH_HANDS_UP"
            elif lUp and self.lhandLmList:
                if lFin == rockPose:  # "land"
                    pose = "LEFT_HAND_ROCKON"
                elif lFin == openPose:  # "left"
                    pose = "LEFT_HAND_OPEN"
                elif lFin == closePose:  # "back"
                    pose = "LEFT_HAND_CLOSE"
            elif rUp and self.rhandLmList:
                if rFin == rockPose:  # "hand_land"
                    pose = "RIGHT_HAND_ROCKON"
                elif rFin == openPose:  # "right"
                    pose = "RIGHT_HAND_OPEN"
                elif rFin == closePose:  # "forward"
                    pose = "RIGHT_HAND_CLOSE"

        return pose, target, distance, tilt


def main():

    detector = PoseModule(face=False)
    fps = FPS()
    cap = cv2.VideoCapture(0)

    while True:

        suc, img = cap.read()

        if suc:

            fps.update(img)

            img_processed = detector.processPose(img)
            img_trimmed = detector.processBBox(img)
            # detector.getLandmarks()
            pose, target, distance, tilt = detector.getInfo()

            cv2.putText(
                img_trimmed,
                str(pose),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                img_trimmed,
                str(target),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                img_trimmed,
                str(distance),
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                img_trimmed,
                str(tilt),
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

            imgList = [img, img_processed, img_trimmed]
            cv2.imshow("Image", stackImages(imgList, 2, 1))

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    main()
