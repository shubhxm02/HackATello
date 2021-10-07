import numpy as np
import tellopy
import logging
from math import atan2, degrees, sqrt
import time
import cv2
import sys
import os
import av
from Utils import FPS

log = logging.getLogger("myTelloPy")


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
        # self.__save_video_feed()
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
        self.drone.set_video_encoder_rate(9)
        retry = 3
        self.container = None
        while self.container is None and 0 < retry:
            retry -= 1
            try:
                self.container = av.open(self.drone.get_video_stream())
            except av.AVError as ave:
                print(ave)
                print("retry...")
        # self.__record_video_feed()

        self.__frame_skip = 600
        for frame in self.container.decode(video=0):
            if self.__frame_skip > 0:
                self.__frame_skip -= 1
                continue
            self.__start_time = time.time()

    def __record_video_feed(self):
        """
        Record video feed
        """
        self.record = True
        self.vid_stream = self.container.streams.video[0]
        self.out_name = os.path.join(
            "recordings", f"tello-video-{get_time_stamp()}.mp4"
        )
        self.out_file = av.open(self.out_name, "w", "mp4")
        self.out_stream = self.out_file.add_stream(
            "mpeg4", self.vid_stream.rate, self.vid_stream.pix_fmt
        )
        self.out_stream.width = self.vid_stream.width
        self.out_stream.height = self.vid_stream.height

    def __save_video_feed(self):
        """
        Save video feed
        """
        self.out_file.close()

    def get_frame(self):
        self.__frame_skip = 0
        for frame in self.container.decode(video=0):
            # if self.__frame_skip > 0:
            #     self.__frame_skip -= 1
            #     continue
            # self.__start_time = time.time()

            # Convert frame to cv2 image
            frame = cv2.cvtColor(
                np.array(frame.to_image(), dtype=np.uint8),
                cv2.COLOR_RGB2BGR,
            )
            frame = cv2.resize(frame, (640, 480))

            # if self.time_base < 1.0 / 60:
            #     self.__time_base = 1.0 / 60
            # else:
            #     self.__time_base = self.time_base
            # self.__frame_skip = int(
            #     (time.time() - self.__start_time) / self.__time_base
            # )

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
            # else:
            #     # This line is necessary to display current values in 'self.write_hud'
            #     self.axis_speed[axis] = self.prev_axis_speed[axis]

    def take_picture(self):
        """
        Tell drone to take picture, image sent to file handler
        """
        self.drone.take_picture()

    def takeoff(self):
        """
        Tell drone to take off
        """
        self.drone.takeoff()

    def throw_and_go(self):
        """
        Tell drone to start a 'throw and go'
        """
        self.drone.throw_and_go()

    def land(self):
        """
        Tell drone to land
        """
        self.drone.land()

    def palm_land(self):
        """
        Tell drone to approach person and land
        """
        log.debug("PALM_LAND")
        self.drone.palm_land()

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

        # if self.write_header_log:
        #     self.write_header_log = False
        #     self.log_file_log.write(f"{data.format_cvs_header()}\n")
        # self.log_file_log.write(f"{data.format_cvs(0)}\n")

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
    fps = FPS()

    tello.connect()
    tello.start_video_feed()

    print("done... stream started")

    while True:
        frame = tello.get_frame()

        fps.update(frame)
        cv2.imshow("frame", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


if __name__ == "__main__":
    main()

    # def set_speed(self, axis, speed):
    #     log.info(f"set speed {axis} {speed}")
    #     self.cmd_axis_speed[axis] = speed

    # def process_frame(self, frame, pose=None, target=None):
    #     """
    #     Analyze the frame and return a control vector.
    #     """
    #     img = frame.copy()
    #     h, w, _ = img.shape
    #     ref_x, ref_y = h // 2, w // 2

    #     # check for scheduled take off
    #     if self.scheduled_takeoff is not None:
    #         if time.time() > self.scheduled_takeoff:
    #             self.drone.takeoff()
    #             self.scheduled_takeoff = None

    #     # if drone is not flying, check for a morse code command
    #     # if not self.is_flying:
    #     #     # add functionality for morse code
    #     #     return frame

    #     # check for scheduled picture
    #     if self.scheduled_picture is not None:
    #         if time.time() > self.scheduled_picture:
    #             self.drone.take_picture()
    #             self.scheduled_picture = None

    #     elif pose is not None:
    #         log.info(f"pose detected : {pose}")
    #         if pose == "BOTH_HANDS_UP":
    #             log.info("take picture in 5 seconds")
    #             self.drone.take_picture()
    #         elif pose == "LEFT_HAND_ROCKON":
    #             log.info("landing")
    #             self.drone.land()
    #         elif pose == "RIGHT_HAND_ROCKON":
    #             log.info("palm_landing")
    #             self.palm_landing_approach = True
    #         elif pose == "LEFT_HAND_OPEN":
    #             log.info("going_left")
    #             self.axis_speed["roll"] = self.def_speed["roll"]
    #         elif pose == "RIGHT_HAND_OPEN":
    #             log.info("going_right")
    #             self.axis_speed["roll"] = -self.def_speed["roll"]
    #         elif pose == "LEFT_HAND_CLOSE":
    #             log.info("going_back")
    #             self.axis_speed["pitch"] = -self.def_speed["pitch"]
    #         elif pose == "RIGHT_HAND_CLOSE":
    #             log.info("going_forward")
    #             self.axis_speed["pitch"] = self.def_speed["pitch"]

    #     if self.tracking:
    #         if target is not None:
    #             cv2.arrowedLine(
    #                 img, (ref_x, ref_y), (target[0], target[1]), (0, 255, 0), 2
    #             )
    #             self.axi

    #     return frame

    # def write_hud(self, frame):
    #     """Draw drone info on frame"""
    #     RED = (0, 0, 255)
    #     BLUE = (255, 0, 0)
    #     GREEN = (0, 255, 0)

    #     frame = frame.copy()

    #     class HUD:
    #         def __init__(self):
    #             self.infos = []

    #         def add(self, info, color=BLUE):
    #             self.infos.append((info, color))

    #         def draw(self, frame):
    #             for i, (info, color) in enumerate(self.infos):
    #                 cv2.putText(
    #                     frame,
    #                     info,
    #                     (0, 30 + (i * 30)),
    #                     cv2.FONT_HERSHEY_SIMPLEX,
    #                     1,
    #                     color,
    #                     thickness=2,
    #                 )

    #     hud = HUD()

    #     hud.add(f"BAT {self.battery}")
    #     hud.add(
    #         f"{'FLYING' if self.is_flying else 'NOT FLYING'}",
    #         GREEN if self.is_flying else RED,
    #     )
    #     hud.add(
    #         f"TRACKING {'ON' if self.tracking else 'OFF'}",
    #         GREEN if self.tracking else RED,
    #     )

    #     if self.axis_speed["yaw"] >= 0:
    #         hud.add(f"CW {self.axis_speed['yaw']}", BLUE)
    #     else:
    #         hud.add(f"CCW {-self.axis_speed['yaw']}", BLUE)
    #     if self.axis_speed["roll"] >= 0:
    #         hud.add(f"RIGHT {self.axis_speed['roll']}", BLUE)
    #     else:
    #         hud.add(f"LEFT {-self.axis_speed['roll']}", BLUE)
    #     if self.axis_speed["pitch"] >= 0:
    #         hud.add(f"FORWARD {self.axis_speed['pitch']}", BLUE)
    #     else:
    #         hud.add(f"BACKWARD {-self.axis_speed['pitch']}", BLUE)
    #     if self.axis_speed["throttle"] >= 0:
    #         hud.add(f"UP {self.axis_speed['throttle']}", BLUE)
    #     else:
    #         hud.add(f"DOWN {-self.axis_speed['throttle']}", BLUE)

    #     hud.draw(frame)
    #     return frame
