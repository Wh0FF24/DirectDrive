# car_controller.py — Serial interface to Arduino for steering, driving, and melodies
import serial
import time


class CarController:
    def __init__(self, port, baud_rate=115200):
        self.port = port
        self.baud_rate = baud_rate
        self.ser = None
        self.current_speed = 0
        self.current_steer = 0

    def connect(self):
        """Connect to the Arduino via serial."""
        try:
            self.ser = serial.Serial(self.port, self.baud_rate)
            self.ser.flushInput()
            print(f"Car connected on {self.port}")
            return True
        except Exception as e:
            print(f"Car not connected: {e}")
            return False

    def disconnect(self):
        """Stop the car and close serial connection."""
        if self.ser and self.ser.is_open:
            self.drive(0)
            time.sleep(0.1)
            self.steer(0)
            self.ser.close()
            print("Car disconnected")

    def _send(self, command):
        """Send a command string to the Arduino."""
        if self.ser and self.ser.is_open:
            self.ser.write((command + "\n").encode())

    def steer(self, angle):
        """Steer the car. angle: -30 (left) to 30 (right)."""
        angle = max(-30, min(30, int(angle)))
        self.current_steer = angle
        self._send(f"s {angle}")

    def drive(self, speed):
        """Set drive speed. speed: -100 to 100 (% throttle)."""
        speed = max(-100, min(100, int(speed)))
        self.current_speed = speed
        self._send(f"d {speed}")

    def stop(self):
        """Stop the car."""
        self.drive(0)

    def play_melody(self, melody_id):
        """Play a melody (0-8)."""
        melody_id = max(0, min(8, int(melody_id)))
        self._send(f"m {melody_id}")

    def emergency_stop(self):
        """Immediate stop — send multiple times for safety."""
        for _ in range(3):
            self.drive(0)
            time.sleep(0.05)
