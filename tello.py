import av
import os
import cv2
import sys
import time
import logging
import numpy as np
from math import atan2
import multiprocessing

import tellopy
from simple_pid import PID
from morse import CameraMorse
from PoseModule import PoseModule
from Utils import FPS, stackImages

log = logging.getLogger("myTelloPy")

HEIGHT, WIDTH = 480, 640
CY, CX = HEIGHT // 2, WIDTH // 2


def get_time_stamp():
    """
    Get time stamp
    """
    cur_time = time.localtime(time.time())
    return time.strftime("%Y%m%d_%H%M%S", cur_time)


class Tello:
    """
    Tello builds keyboard controls on top of TelloPy
    as well as generationg images from video stream.
    """

    def __init__(self):

        path = os.path.join("logs", f"tello-{get_time_stamp()}.csv")
        path_log = os.path.join("logs", f"tello-log-{get_time_stamp()}.csv")
        self.log_file = open(path, "w")
        self.log_file_log = open(path_log, "w")
        self.write_header = True
        self.write_header_log = True
        log.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        log.addHandler(ch)

        self.drone = tellopy.Tello()
        self.reset()

        self.flight_data = None
        self.prev_flight_data = None

        self.is_flying = False
        self.fly_mode = None
        self.battery = None
        self.throw_fly_timer = 0
        self.throw_ongoing = False
        self.tracking_after_takeoff = True

        self.axis_speed = {"yv": 0, "rl": 0, "fb": 0, "ud": 0}
        self.cmd_axis_speed = {"yv": 0, "rl": 0, "fb": 0, "ud": 0}
        self.prev_axis_speed = self.axis_speed.copy()
        self.def_speed = {"yv": 50, "rl": 35, "fb": 35, "ud": 80}

        self.axis_command = {
            "yv": self.drone.clockwise,
            "rl": self.drone.right,
            "fb": self.drone.forward,
            "ud": self.drone.up,
        }

        # yv : yaw, rl : roll, fb : pitch, ud : throttle
        self.speed_def = {"yv": 50, "rl": 30, "fb": 30, "ud": 30}
        self.speed_curr = {"yv": 0, "rl": 0, "fb": 0, "ud": 0}
        self.speed_prev = self.speed_curr.copy()
        self.speed_null = self.speed_curr.copy()

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

        self.stream_raw = None
        self.stream_pro = None
        self.stream_trm = None
        self.stream = None

    def reset(self):
        """
        Tell drone to reset
        """
        log.debug("RESET")
        self.ref_pos_x = -1
        self.ref_pos_y = -1
        self.ref_pos_z = -1
        self.pos_x = -1
        self.pos_y = -1
        self.pos_z = -1
        self.yaw = 0
        self.throw_ongoing = False

    def quit(self):
        """
        Tell drone to land and terminate processes
        """
        log.debug("QUIT")
        self.drone.land()
        self.drone.quit()

    def connect(self):
        """
        Connect to drone
        """
        self.reset()
        self.drone.connect()
        self.drone.wait_for_connection(60.0)
        self.drone.subscribe(self.drone.EVENT_LOG_DATA, self.__log_data_handler)
        self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA, self.__flight_data_handler)
        self.drone.subscribe(
            self.drone.EVENT_FILE_RECEIVED, self.__handle_file_received
        )
        # self.send_rc_commands(0, 0, 0, 0)

    def start_video_feed(self):
        """
        Start video feed
        """
        self.drone.start_video()
        # self.drone.set_video_encoder_rate(9)
        retry = 3
        self.container = None
        while self.container is None and 0 < retry:
            retry -= 1
            try:
                self.container = av.open(self.drone.get_video_stream())
            except av.AVError as ave:
                print(ave)
                print("retry...")

        self.__frame_skip = 300
        for frame in self.container.decode(video=0):
            if self.__frame_skip > 0:
                self.__frame_skip -= 1
                continue
            else:
                break

    def get_stream(self):
        for frame in self.container.decode(video=0):
            # Convert frame to cv2 image
            frame = cv2.cvtColor(
                np.array(frame.to_image(), dtype=np.uint8),
                cv2.COLOR_RGB2BGR,
            )
            frame = cv2.resize(frame, (640, 480))

            return frame

    def get_stream_matrix(self):
        """
        returns the raw stream from the drone
        """
        # try:
        self.stream_raw = self.frame.copy()
        self.fps = self.FPS.update(self.stream_raw)

        _img_list = [self.stream_raw, self.stream_pro, self.stream_trm, self.stream]
        # except:
        #     return None
        # return stackImages(_img_list, 2, 1)
        return self.stream

    def process_stream(self, frame):

        frame = frame.copy()

        stream_pro = self.pd.processPose(frame)
        stream_trm = self.pd.processBBox(frame)

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

        if self.stream_raw is None:
            return None

        if not self.is_flying:
            morse_code = self.morse.eval(self.stream_raw)
            if morse_code == "...":
                self.scheduled_throwgo = True
            elif morse_code == "---":
                self.scheduled_takeoff = True
        elif self.battery < 20:
            self.set_speed(self.speed_null)
            self.scheduled_landing = time.time() + 10

        if self.scheduled_takeoff:
            self.set_speed(self.speed_null)
            if self.scheduled_takeoff > time.time():
                self.drone.takeoff()
                self.tracking = True
                self.scheduled_takeoff = None

        elif self.scheduled_throwgo:
            self.set_speed(self.speed_null)
            if self.scheduled_throwgo > time.time():
                self.drone.takeoff()
                self.tracking = True
                self.scheduled_throwgo = None

        elif self.scheduled_picture:
            self.set_speed(self.speed_null)
            if self.scheduled_picture > time.time():
                self.drone.take_picture()
                self.tracking = True
                self.scheduled_picture = None

        elif self.scheduled_landing:
            self.set_speed(self.speed_null)
            if self.scheduled_landing > time.time():
                self.drone.land()
                self.tracking = False
                self.scheduled_landing = None

        elif self.scheduled_palmland:
            self.set_speed(self.speed_null)
            if distance > 270:
                self.drone.palm_land()
                self.tracking = False
                self.scheduled_palmland = None
            else:
                self.speed_curr["fb"] = 20
                self.set_speed(self.speed_curr)

        elif self.tracking == True and target is None:
            # rotate to find target
            self.searching = True
            self.speed_curr["yv"] = 30
            self.set_speed(self.speed_curr)

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
            self.set_speed(self.speed_curr)

        stream = self.write_hud()

        self.speed_curr = self.speed_null.copy()

        return stream

    def write_hud(self, frame):
        """
        writes the hud to the frame
        """

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
        hud.add(f"BAT {self.battery}", PURPLE)
        hud.add(
            f"{'FLYING' if self.is_flying else 'NOT_FLYING'}",
            GREEN if self.is_flying else RED,
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

        frame = frame.copy()
        hud.draw(frame)
        return frame

    def set_speed(self, axis_speed={"yv": 0, "rl": 0, "fb": 0, "ud": 0}):
        self.axis_speed["yv"] = axis_speed["yv"]
        self.axis_speed["ud"] = axis_speed["ud"]
        self.axis_speed["rl"] = axis_speed["rl"]
        self.axis_speed["fb"] = axis_speed["fb"]
        for axis, command in self.axis_command.items():
            if (
                self.axis_speed[axis] is not None
                and self.axis_speed[axis] != self.prev_axis_speed[axis]
            ):
                log.debug(f"COMMAND {axis} : {self.axis_speed[axis]}")
                command(self.axis_speed[axis])
                self.prev_axis_speed[axis] = self.axis_speed[axis]

    def __handle_file_received(self, event, sender, data):
        """
        Create a file in local directory to receive image from the drone
        """
        path = os.path.join("images", f"tello-{get_time_stamp()}.jpg")
        with open(path, "wb") as out_file:
            out_file.write(data)
        print(f"Saved photo to {path}")

    def __flight_data_handler(self, event, sender, data):
        """
        Handler for flight data events
        """
        self.battery = data.battery_percentage
        self.fly_mode = data.fly_mode
        self.throw_fly_timer = data.throw_fly_timer
        self.throw_ongoing = data.throw_fly_timer > 0

        if self.prev_flight_data != str(data):
            print(data)
            self.prev_flight_data = str(data)
        self.flight_data = data

        if self.is_flying != data.em_sky:
            self.is_flying = data.em_sky
            log.debug(f"FLYING : {self.is_flying}")
            if not self.is_flying:
                self.reset()
            else:
                if self.tracking_after_takeoff:
                    log.info("Tracking on after takeoff")
                    self.toggle_tracking(True)

    def __log_data_handler(self, event, sender, data):
        """
        Listener to log data from the drone.
        """
        pos_x = -data.mvo.pos_x
        pos_y = -data.mvo.pos_y
        pos_z = -data.mvo.pos_z
        # First time we have meaningful values, we store them as reference
        if abs(pos_x) + abs(pos_y) + abs(pos_z) > 0.07:
            if self.ref_pos_x == -1:
                self.ref_pos_x = pos_x
                self.ref_pos_y = pos_y
                self.ref_pos_z = pos_z
            else:
                self.pos_x = pos_x - self.ref_pos_x
                self.pos_y = pos_y - self.ref_pos_y
                self.pos_z = pos_z - self.ref_pos_z

        qx = data.imu.q1
        qy = data.imu.q2
        qz = data.imu.q3
        qw = data.imu.q0

        degree = 0.01745
        siny = 2 * (qw * qz + qx * qy)
        cosy = 1 - 2 * (qy * qy + qz * qz)
        self.yaw = int(atan2(siny, cosy) / degree)

        if self.write_header:
            self.log_file.write(f"{data.format_cvs_header()}\n")
            self.write_header = False
        self.log_file.write(f"{data.format_cvs()}\n")


def main():
    """
    Main function
    """
    tello = Tello()

    tello.connect()
    tello.start_video_feed()

    # video1 = cv2.VideoWriter(
    #     os.path.join("recordings", f"{get_time_stamp()}raw.mp4"),
    #     cv2.VideoWriter_fourcc(*"XVID"),
    #     30,
    #     (WIDTH, HEIGHT),
    # )
    video2 = cv2.VideoWriter(
        os.path.join("recordings", f"{get_time_stamp()}.avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        30,
        (WIDTH, HEIGHT),
    )

    start = False

    print("done... stream started")
    # time.sleep(15)
    print("streaming now")

    img = None
    p1 = multiprocessing.Process(target=tello.get_stream)
    p2 = multiprocessing.Process(target=tello.process_stream, args=(img,))

    while True:

        # p1.start()
        # p2.start()

        # p1.join()
        # p2.join()

        raw = tello.get_stream()
        cv2.imshow("img", raw)

        if start:
            fin = tello.process_stream(cv2.resize(raw, (480, 360)))
            try:
                cv2.imshow("tello", fin)
            except Exception as e:
                print(e)

        # stream = tello.get_stream_matrix()
        # video1.write(stream)

        video2.write(raw)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("s"):
            start = True
        if k == 27:
            # video1.release()
            video2.release()
            tello.quit()
            break


if __name__ == "__main__":
    main()
