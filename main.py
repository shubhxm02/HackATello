import cv2
import time
from simple_pid import PID
from mytellopy import Tello

from morse import CameraMorse
from PoseModule import PoseModule
from Utils import FPS, stackImages

TESTING = 1
# HEIGHT, WIDTH = 360, 480
HEIGHT, WIDTH = 180, 240
CY, CX = HEIGHT // 2, WIDTH // 2


class TelloController:
    """
    TelloController builds velocity controls
    and pose commands over mytellopy class
    """

    def __init__(self, tracking=True):

        # yv : yaw, rl : roll, fb : pitch, ud : throttle
        self.speed_def = {"yv": 50, "rl": 30, "fb": 30, "ud": 30}
        self.speed_curr = {"yv": 0, "rl": 0, "fb": 0, "ud": 0}
        self.speed_prev = self.speed_curr.copy()
        self.speed_null = self.speed_curr.copy()

        self.drone = Tello()
        self.drone.connect()
        self.drone.start_video_feed()

        self.pd = PoseModule(face=False)
        self.FPS = FPS()

        self.tracking = None
        self.searching = None
        self.scheduled_picture = None
        self.scheduled_takeoff = None
        self.scheduled_throwgo = None
        self.scheduled_landing = None
        self.scheduled_palmland = None

        self.tilt = -1
        self.distance = 300
        self.pose = None

        self.pid_speed_fb = PID(
            Kp=0.1,
            Ki=0.1,
            Kd=0.1,
            setpoint=self.distance,
            sample_time=0,
            output_limits=(-100, 100),
            auto_mode=True,
        )
        self.pid_speed_yv = PID(
            Kp=0.1,
            Ki=0.1,
            Kd=0.1,
            setpoint=CX,
            sample_time=0,
            output_limits=(-100, 100),
            auto_mode=True,
        )
        self.pid_speed_ud = PID(
            Kp=0.1,
            Ki=0.1,
            Kd=0.1,
            setpoint=CY,
            sample_time=0,
            output_limits=(-100, 100),
            auto_mode=True,
        )

        self.morse = CameraMorse()
        # self.morse.define_command("---", self.scheduled_takeoff)
        # self.morse.define_command("...", self.scheduled_throwgo)

    def get_stream(self):
        """
        returns the raw stream from the drone
        """
        self.fps = self.FPS.update()
        self.stream_raw = self.drone.get_frame()
        self.stream_raw = cv2.resize(self.stream_raw, (WIDTH, HEIGHT))
        self.stream_pro = self.pd.processPose(self.stream_raw)
        self.stream_trm = self.pd.processBBox(self.stream_raw)

    def process_stream(self):
        pose, target, distance, tilt = self.pd.getInfo()

        try:
            self.pose = pose
            tilt = (tilt // 15) * 15
            distance = (distance // 10) * 10
            target[0] = (target[0] // 10) * 10
            target[1] = (target[1] // 10) * 10
        except:
            tilt = None
            distance = None

        if not self.drone.is_flying:
            morse_code = self.morse.eval(self.stream_raw)
            if morse_code == "...":
                self.scheduled_throwgo = True
            elif morse_code == "---":
                self.scheduled_takeoff = True
        elif self.drone.battery < 20:
            self.drone.set_speed(self.speed_null)
            self.scheduled_landing = time.time() + 10

        if self.scheduled_takeoff:
            self.drone.set_speed(self.speed_null)
            if self.scheduled_takeoff > time.time():
                self.drone.takeoff()
                self.tracking = True
                self.scheduled_takeoff = None

        elif self.scheduled_throwgo:
            self.drone.set_speed(self.speed_null)
            if self.scheduled_throwgo > time.time():
                self.drone.takeoff()
                self.tracking = True
                self.scheduled_throwgo = None

        elif self.scheduled_picture:
            self.drone.set_speed(self.speed_null)
            if self.scheduled_picture > time.time():
                self.drone.take_picture()
                self.tracking = True
                self.scheduled_picture = None

        elif self.scheduled_landing:
            self.drone.set_speed(self.speed_null)
            if self.scheduled_landing > time.time():
                self.drone.land()
                self.tracking = False
                self.scheduled_landing = None

        elif self.scheduled_palmland:
            self.drone.set_speed(self.speed_null)
            if distance > 270:
                self.drone.palm_land()
                self.tracking = False
                self.scheduled_palmland = None
            else:
                self.speed_curr["fb"] = 20
                self.drone.set_speed(self.speed_curr)

        elif self.tracking == True and target is None:
            # rotate to find target
            self.searching = True
            self.speed_curr["yv"] = 30
            self.drone.set_speed(self.speed_curr)

        elif self.tracking == True and target is not None:

            self.searching = False

            # use pid to keep target at center
            self.speed_curr["yv"] = self.pid_speed_yv(target[0])
            self.speed_curr["ud"] = self.pid_speed_ud(target[1])

            # update defalut distance when tilt changes
            if self.tilt != tilt:
                self.distance = distance
                self.tilt = tilt
                self.pid_speed_fb.setpoint = self.distance

            # use pid to maintain distance
            self.speed_curr["fb"] = self.pid_speed_fb(distance)

            # pose control
            if pose is None:
                self.speed_curr["rl"] = 0
            elif pose == "BOTH_HANDS_UP":
                self.scheduled_picture = time.time() + 3
            elif pose == "LEFT_HAND_ROCKON":
                self.scheduled_landing = time.time() + 1
            elif pose == "RIGHT_HAND_ROCKON":
                self.scheduled_palmland = time.time() + 1
            elif pose == "LEFT_HAND_OPEN":
                self.speed_curr["rl"] = self.speed_def["rl"]
            elif pose == "RIGHT_HAND_OPEN":
                self.speed_curr["rl"] = -self.speed_def["rl"]
            elif pose == "LEFT_HAND_CLOSE":
                self.tilt = -1
                self.speed_curr["fb"] = -self.speed_def["fb"]
            elif pose == "RIGHT_HAND_CLOSE":
                self.tilt = -1
                self.speed_curr["fb"] = self.speed_def["fb"]

            # send speed to drone
            self.drone.set_speed(self.speed_curr)

        self.frame = self.write_hud(self.stream_raw)

        self.speed_curr = self.speed_null.copy()

        return self.frame

    def write_hud(self, frame):
        """
        writes the hud to the frame
        """
        frame = frame.copy()

        RED = (0, 0, 255)
        GREEN = (0, 255, 0)
        BLUE = (255, 0, 0)
        PURPLE = (128, 0, 128)
        CYAN = (0, 255, 255)
        YELLOW = (255, 255, 0)

        class HUD:
            def __init__(self):
                self.infos = []

            def add(self, info, color=CYAN):
                self.infos.append((info, color))

            def draw(self, frame):
                i = 0
                for i, (info, color) in enumerate(self.infos):
                    cv2.putText(
                        frame,
                        info,
                        (0, 30 + (i * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        color,
                        2,
                    )  # lineType=30)

        hud = HUD()

        hud.add(f"FPS {self.fps}", PURPLE)
        hud.add(f"BAT {self.drone.battery}", PURPLE)
        hud.add(
            f"{'FLYING' if self.drone.is_flying else 'NOT_FLYING'}",
            GREEN if self.drone.is_flying else RED,
        )
        hud.add(
            f"TRACKING {'ON' if self.tracking else 'OFF'}",
            GREEN if self.tracking else RED,
        )

        if self.speed_curr["yv"] >= 0:
            hud.add(f"CW {self.speed_curr['yv']}", GREEN)
        elif self.speed_curr["yv"] < 0:
            hud.add(f"CCW {-self.speed_curr['yv']}", RED)
        if self.speed_curr["rl"] >= 0:
            hud.add(f"RIGHT {self.speed_curr['rl']}", GREEN)
        elif self.speed_curr["rl"] < 0:
            hud.add(f"LEFT {-self.speed_curr['rl']}", RED)
        if self.speed_curr["fb"] >= 0:
            hud.add(f"FORWARD {self.speed_curr['fb']}", GREEN)
        elif self.speed_curr["fb"] < 0:
            hud.add(f"BACKWARD {-self.speed_curr['fb']}", RED)
        if self.speed_curr["ud"] >= 0:
            hud.add(f"UP {self.speed_curr['ud']}", GREEN)
        elif self.speed_curr["ud"] < 0:
            hud.add(f"DOWN {-self.speed_curr['ud']}", RED)

        hud.add(f"POSE: {self.pose}", GREEN if self.pose else RED)
        hud.add(f"distance: {self.distance}", BLUE)
        hud.add(f"tilt: {self.tilt}", BLUE)

        if self.scheduled_picture:
            hud.add(f"picture in: {self.scheduled_picture - time.time()}")
        if self.scheduled_landing:
            hud.add(f"landing in: {self.scheduled_landing - time.time()}")
        if self.scheduled_palmland:
            hud.add(f"palmland in: {self.scheduled_palmland - time.time()}")
        if self.scheduled_throwgo:
            hud.add(f"throw in: {self.scheduled_throwgo - time.time()}")
        if self.scheduled_takeoff:
            hud.add(f"takeoff in: {self.scheduled_takeoff - time.time()}")
        if self.searching:
            hud.add("Searching for person..", PURPLE)

        hud.draw(frame)
        return frame


def main():

    # detector = PoseModule(face=False)
    # fps = FPS()
    drone = TelloController()

    print("///////////////////////////")
    print("started streaming from main")
    print("///////////////////////////")
    while True:

        drone.get_stream()
        img = drone.process_stream()

        # cv2.putText(
        #     img_trimmed,
        #     str(pose) + str(target) + str(distance),
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (0, 0, 255),
        #     2,
        # )

        # imgList = [img, img_processed, img_trimmed]
        # cv2.imshow("Image", stackImages(imgList, 2, 1))

        cv2.imshow("img", img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
