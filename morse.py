import numpy as np
import cv2
from time import time, sleep


class RollingGraph:
    """
    Class designed to draw in an OpenCv window, graph of variables evolving with time.
    The time is not absolute here, but each new call to method 'new_iter' corresponds to a time step.
    'new_iter' takes as argument an array of the current variable values
    """

    def __init__(
        self,
        window_name="Graph",
        width=640,
        height=250,
        step_width=5,
        y_min=0,
        y_max=255,
        colors=[(0, 0, 255)],
        thickness=[2],
        threshold=None,
        waitKey=True,
    ):
        """
        width, height: width and height in pixels of the OpenCv window in which the graph is draw
        step_width: width in pixels on the x-axis between each 2 consecutive points
        y_min, y_max : min and max of the variables
        colors : array of the colors used to draw the variables
        thickness: array of the thickness of the variable curves
        waitKey : boolean. In OpenCv, to display a window, we must call cv2.waitKey(). This call can be done by RollingGraph (if True) or by the program who calls RollingGraph (if False)
        """
        self.window_name = window_name
        self.width = width
        self.height = height
        self.step_width = step_width
        self.y_min = y_min
        self.y_max = y_max
        self.waitKey = waitKey
        assert len(colors) == len(thickness)
        self.colors = colors
        self.thickness = thickness
        self.iter = 0
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.nb_values = len(colors)
        self.threshold = threshold

    def new_iter(self, values):
        # Values = array of values, same length as colors
        assert len(values) == self.nb_values
        self.iter += 1
        if self.iter > 1:
            if self.iter * self.step_width >= self.width:
                self.canvas[:, 0 : self.step_width, :] = 0
                self.canvas = np.roll(self.canvas, -self.step_width, axis=1)
                self.iter -= 1
            for i in range(self.nb_values):
                cv2.line(
                    self.canvas,
                    (
                        (self.iter - 1) * self.step_width,
                        int(
                            self.height
                            - self.prev_values[i]
                            * self.height
                            / (self.y_max - self.y_min)
                        ),
                    ),
                    (
                        self.iter * self.step_width,
                        int(
                            self.height
                            - values[i] * self.height / (self.y_max - self.y_min)
                        ),
                    ),
                    self.colors[i],
                    self.thickness[i],
                )
            if self.threshold:
                cv2.line(
                    self.canvas,
                    (
                        0,
                        int(
                            self.height
                            - self.threshold * self.height / (self.y_max - self.y_min)
                        ),
                    ),
                    (
                        self.width,
                        int(
                            self.height
                            - self.threshold * self.height / (self.y_max - self.y_min)
                        ),
                    ),
                    (0, 255, 0),
                    1,
                )
            cv2.imshow(self.window_name, self.canvas)
            if self.waitKey:
                cv2.waitKey(1)
        self.prev_values = values


class CameraMorse:
    """
    Designed with the Tello drone in mind but could be used with other small cameras.
    When the Tello drone is not flying, we can use its camera as a way to pass commands to the calling script.
    Covering/uncovering the camera with a finger, is like pressing/releasing a button.
    Covering/uncovering the camera is determined by calculating the level of brightness of the frames received from the camera
    Short press = dot
    Long press = dash
    If we associate series of dots/dashes to commands, we can then ask the script to launch these commands.
    """

    def __init__(self):
        """
        display : True to display to display brightness(time) in an opencv window (via an object RollingGraph)
        """
        # Durations below are in seconds
        # 0 < duration of a dot <= dot_duration
        self.dot_duration = 0.3
        # dot_duration < duration of a dash <= dash_duration
        self.dash_duration = 3 * self.dot_duration
        # Released duration.
        self.blank_duration = 5

        # Dots or dashes are delimited by a "press" action followed by a "release" action
        # In normal situation, the brightness is above 'threshold'
        # When brightness goes below 'threshold' = "press" action
        # Then when brightness goes back above 'threshold' = "release" action
        self.threshold = 40

        # Dictionary that associates codes to commands
        self.dot_commands = ["...", "..-", "-..", ".-."]
        self.dash_commands = ["---", "-.-", ".--", "--."]

        # Current status
        self.is_pressed = False

        # Timestamp of the last status change (pressed/released)
        self.timestamp = 0

        # Current morse code. String composed of '.' and '-'
        self.code = ""

        # self.graph_brightness = RollingGraph(threshold=self.threshold)

    def eval(self, frame):
        """
        Analyze the frame 'frame', detect potential 'dot' or 'dash', and if so, check
        if we get a defined code
        Returns:
        - a boolean which indicates if the "button is pressed" or not,
        - "dot" or "dash"  if a dot or a dash has just been detected, or None otherwise
        """
        self.brightness = np.mean(frame)
        pressing = self.brightness < self.threshold
        current_time = time()
        # if self.display:
        # self.graph_brightness.new_iter([self.brightness])

        if self.is_pressed and not pressing:  # Releasing
            if (
                current_time - self.timestamp > self.blank_duration
            ):  # The press was too long, we cancel the current decoding
                self.code = ""
            else:
                self.is_pressed = False
                self.timestamp = current_time
            if current_time - self.timestamp < self.dot_duration:  # We have a dot
                self.code += "."
            elif current_time - self.timestamp < self.dash_duration:  # We have a dash
                self.code += "-"
        elif not self.is_pressed and pressing:  # Pressing
            if (
                current_time - self.timestamp > self.blank_duration
            ):  # The blank was too long, we cancel the current decoding
                self.code = ""
            self.is_pressed = True
            self.timestamp = current_time

        if self.code in self.dot_commands:
            self.code = ""
            return "..."
        elif self.code in self.dash_commands:
            self.code = ""
            return "---"
        else:
            return ""


if __name__ == "__main__":

    def test(arg=1):
        print("Function test:", arg)

    frame = {"w": 220 * np.ones((10, 10)), "b": 20 * np.ones((10, 10))}

    frames = [
        "w",
        "w",
        "w",
        "b",
        "w",
        "b",
        "w",
        "w",
        "w",
        "w",
        "w",
        "b",
        "w",
        "b",
        "b",
        "w",
        "w",
        "w",
    ]

    cm = CameraMorse(display=True)
    cm.define_command("..", test)
    cm.define_command(".-", test, {"arg": 2})

    for f in frames:
        print(cm.eval(frame[f]))
        sleep(0.10)
    cv2.waitKey(0)
