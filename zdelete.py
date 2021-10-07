# testing out whether pose estimation works
# on raw video feed from the tello drone

import multiprocessing
import av
import os
import cv2
import time
import numpy as np

import tellopy
from PoseModule import PoseModule
from Utils import FPS, stackImages

from multiprocessing import Process, Pipe, sharedctypes


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
        self.drone = tellopy.Tello()

        self.pd = PoseModule(face=False)
        self.FPS = FPS()

        self.container = None

    def quit(self):
        """
        Tell drone to land and terminate processes
        """
        self.drone.land()
        self.drone.quit()

    def connect(self):
        """
        Connect to drone
        """
        self.drone.connect()
        self.drone.wait_for_connection(60.0)
        # self.drone.subscribe(self.drone.EVENT_LOG_DATA, self.__log_data_handler)
        # self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA, self.__flight_data_handler)
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

    def process_stream(self, frame):

        img = frame.copy()

        self.FPS.update(img=img)

        img_pro = self.pd.processPose(img)
        img_tri = self.pd.processBBox(img)

        imgList = [img, img_pro, img_tri]
        final = stackImages(imgList, 2, 1)

        return final

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
            if not self.is_flying:
                self.reset()
            else:
                if self.tracking_after_takeoff:
                    self.toggle_tracking(True)


tello2 = Tello()


def openpose_worker():
    """
    In 2 processes mode, this is the init and main loop of the child
    """
    while True:
        # tello.fps.update()

        frame = np.ctypeslib.as_array(tello2.shared_array).copy()
        frame.shape = tello2.frame_shape

        final = tello2.process_stream(frame)

        cv2.imshow("Processed", final)

        cv2.waitKey(1)


def main():
    """
    Main function
    """

    tello2.connect()
    tello2.start_video_feed()

    frame_skip = 0
    start_time = 0
    time_base = 0
    first_frame = True
    frame = None
    multiprocessing = False

    parent_cnx, child_cnx = Pipe()

    for frame in tello2.container.decode(video=0):
        if 0 < frame_skip:
            frame_skip = frame_skip - 1
            continue
        start_time = time.time()
        if frame.time_base < 1.0 / 60:
            time_base = 1.0 / 60
        else:
            time_base = frame.time_base

        # Convert frame to cv2 image
        frame = cv2.cvtColor(
            np.array(frame.to_image(), dtype=np.uint8), cv2.COLOR_RGB2BGR
        )
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.resize(frame, (320, 240))

        if multiprocessing:
            if first_frame:
                # Create the shared memory to share the current frame decoded by the parent process
                # and given to the child process for further processing (openpose, write_hud,...)
                frame_as_ctypes = np.ctypeslib.as_ctypes(frame)
                tello2.shared_array = sharedctypes.RawArray(
                    frame_as_ctypes._type_, frame_as_ctypes
                )
                tello2.frame_shape = frame.shape
                first_frame = False
                # Launch process child
                p_worker = Process(target=openpose_worker)
                p_worker.start()
            # Write the current frame in shared memory
            tello2.shared_array[:] = np.ctypeslib.as_ctypes(frame.copy())
            # Check if there is some message from the child
            if parent_cnx.poll():
                msg = parent_cnx.recv()
                if msg == "EXIT":
                    print("MAIN EXIT")
                    p_worker.join()
                    tello2.drone.quit()
                    cv2.destroyAllWindows()
                    exit(0)

        else:
            final = tello2.process_stream(frame)
            cv2.imshow("pro", final)

        cv2.imshow("raw", frame)

        frame_skip = int((time.time() - start_time) / time_base)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            tello2.quit()
            break


if __name__ == "__main__":
    main()
