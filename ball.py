import math
import numpy as np


class BallBase:
    r = 0
    x = 0
    y = 0
    key = -1
    type = -1
    color = (255, 0, 0)


class Ball(BallBase):
    x = 0
    y = 0
    r = 25.6
    key = -1
    type = -1
    alt = []
    color = (255, 0, 0)

    @property
    def center(self):
        return (self.x, self.y)

    @property
    def array(self):
        return [self.x, self.y, self.r]

    @property
    def intX(self):
        return int(self.x)

    @property
    def intY(self):
        return int(self.y)

    @property
    def intR(self):
        return int(self.r)

    @property
    def intCenter(self):
        return (self.intX, self.intY)

    @property
    def intArray(self):
        return [self.intX, self.intY, self.intR]

    @property
    def keyChr(self):
        return chr(self.key)

    @property
    def upperKeyChr(self):
        return chr(self.key).upper()

    def __init__(self, x: float, y: float, r: float = None) -> None:
        self.x = x
        self.y = y
        self.r = r if r else Ball.r

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Ball):
            return self.x == __o.x and self.y == __o.y and self.r == __o.r
        return False

    def __str__(self) -> str:
        return f"Ball(x: {self.x}, y: {self.y}, r: {self.r}, key: {self.key}, type: {self.type}, color: {self.color})"

    def distance(self, other: BallBase):
        return math.sqrt((other.x - self.x) ** 2 + (other.y - self.y) ** 2)

    def angle_range_dist(self, other: BallBase):
        # right 0, down 90, left 180, up 270
        angle = math.atan2(other.y - self.y, other.x - self.x) % (2 * math.pi)
        distance = self.distance(other)
        range = math.asin((other.r + self.r) / distance)
        return angle, range, distance

    def aabs(self, min_angle, max_angle):
        if max_angle < min_angle:
            max_angle += 2 * math.pi
        return max_angle

    def find_reachable_balls(self, balls: list[BallBase]):
        """
        return `list` of (`ball`, `min_angle`, `max_angle`, `distance`)
        """
        all_balls: list[Ball] = [b for b in balls if b != self]

        angle_ranges = []
        for ball in all_balls:
            (angle, range, dist) = self.angle_range_dist(ball)
            min_angle = (angle - range + 2 * math.pi) % (2 * math.pi)
            max_angle = self.aabs(min_angle, angle + range)
            angle_ranges.append((ball, min_angle, max_angle, dist))
        angle_ranges.sort(key=lambda x: x[3])

        available_angles = []

        for ball, min_angle, max_angle, dist in angle_ranges:
            conflicts = [
                a
                for a in [b for b in angle_ranges if b[3] < dist]
                if a[1] <= min_angle <= a[2] or a[1] <= max_angle <= a[2]
            ]

            # if no conflict at all
            if len(conflicts) == 0:
                available_angles.append((ball, min_angle, max_angle, dist))
                continue

            # calculates remaining available angle range

            for c in conflicts:
                c_min = c[1]
                c_max = c[2]
                if c_min <= min_angle <= c_max:
                    min_angle = c_max + 0.0001
                if c_min <= max_angle <= c_max:
                    max_angle = c_min - 0.0001

            if min_angle <= max_angle:
                available_angles.append((ball, min_angle, max_angle, dist))

        return np.array(available_angles)
