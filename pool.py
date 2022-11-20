import math
import random
from time import sleep

import cv2
import numpy as np
import pyautogui as auto
from ball import Ball

import keyboard as kb


class Pool:

    x = None
    y = None
    w = None
    h = None

    left = None
    right = None
    top = None
    bottom = None

    balls: list[Ball] = []

    table_lb = np.array([50, 140, 0])
    table_ub = np.array([65, 180, 255])

    def screenshot(self):
        sc = auto.screenshot()
        return np.array(sc)

    def create_table_mask(self, img_bgr: cv2.Mat):
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        # Find table area (green color mask)
        return cv2.inRange(img_hsv, self.table_lb, self.table_ub)

    def find_cue_ball(self, img_bgr: cv2.Mat, balls: list[Ball]):
        img_cropped = img_bgr[self.y : self.y + self.h, self.x : self.x + self.w]
        img_balls_hsv = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2HSV)
        # Find color of cue ball
        # hsv(40, 5%, 65%) - hsv(80, 15%, 100%)
        # cv2.hsv (100, 14, 166) - (80, 38, 255)
        lb = np.array([70, 20, 100])
        ub = np.array([120, 60, 255])
        mask_cue = cv2.inRange(img_balls_hsv, lb, ub)
        raw_contours, _ = cv2.findContours(
            mask_cue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        threshold = 1900
        contours = []
        contour = None
        while len(contours) == 0:
            contours = [c for c in raw_contours if cv2.contourArea(c) > threshold]
            threshold -= 100
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        contours = [contours[max_index]]
        contour = contours[0]
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w / 2, y + h / 2)
        gballs = self.intersected_circles(center, balls)
        cue_ball = gballs[0]
        cue_ball.type = 1
        cue_ball.color = (255, 0, 255)

        return cue_ball

    def find_ghost(self, img_bgr: cv2.Mat, balls: list[Ball]):
        img_cropped = img_bgr[self.y : self.y + self.h, self.x : self.x + self.w].copy()
        mask_balls = np.zeros(img_cropped.shape[:2], dtype=np.uint8)
        for ball in balls:
            cv2.circle(mask_balls, ball.intCenter, ball.intR, 255, -1)
        img_balls = cv2.bitwise_and(img_cropped, img_cropped, mask=mask_balls)
        img_balls_hsv = cv2.cvtColor(img_balls, cv2.COLOR_BGR2HSV)

        mask_ghost = cv2.inRange(img_balls_hsv, self.table_lb, self.table_ub)
        raw_contours, _ = cv2.findContours(
            mask_ghost, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = [c for c in raw_contours if cv2.contourArea(c) > 400]
        areas = [cv2.contourArea(c) for c in contours]

        if len(contours) == 0:
            return None

        max_index = np.argmax(areas)
        contour = contours[max_index]

        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w / 2, y + h / 2)
        ghost_ball = self.intersected_circles(center, balls)[0]
        return ghost_ball

    def find_table_area(self, img_bgr: cv2.Mat, table_mask=None):

        table_mask = (
            self.create_table_mask(img_bgr) if table_mask is None else table_mask
        )

        # Find table area shape (x, y, w, h)
        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Find the biggest contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]

        # Table location and size
        self.x, self.y, self.w, self.h = cv2.boundingRect(cnt)
        self.find_table_bounds(img_bgr)

    # https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv
    def find_table_bounds(self, img_bgr: cv2.Mat):
        img_cropped = img_bgr[self.y : self.y + self.h, self.x : self.x + self.w]
        img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        blur_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
        low_threshold = 30
        high_threshold = 40
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 400  # minimum number of pixels making up a line
        max_line_gap = 10  # maximum gap in pixels between connectable line segments

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(
            edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap
        )

        # line_image = img_bgr.copy()
        # for line in lines:
        #     for x1, y1, x2, y2 in line:
        #         cv2.line(
        #             line_image,
        #             (x1 + self.x, y1 + self.y),
        #             (x2 + self.x, y2 + self.y),
        #             (255, 0, 255),
        #             1,
        #         )

        # cv2.imshow("Table", cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # vertical
        vertical = [
            [[x1, y1, x2, y2]]
            for [[x1, y1, x2, y2]] in lines
            if abs(y2 - y1) > abs(x2 - x1)
        ]
        vertical.sort(key=lambda x: x[0][0])
        left = vertical[1][0][0] + self.x
        right = vertical[-2][0][0] + self.x

        # horizontal
        horizontal = [
            [[x1, y1, x2, y2]]
            for [[x1, y1, x2, y2]] in lines
            if abs(y2 - y1) < abs(x2 - x1)
        ]
        horizontal.sort(key=lambda x: x[0][1])
        top = horizontal[1][0][1] + self.y
        bottom = horizontal[-2][0][1] + self.y

        line_image = img_bgr.copy()
        # cv2.rectangle(line_image, (left, top), (right, bottom), (255, 0, 0), 1)
        cv2.imshow(
            "Table bounds",
            cv2.cvtColor(line_image[top:bottom, left:right], cv2.COLOR_BGR2RGB),
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.x = left
        self.y = top
        self.w = right - left
        self.h = bottom - top
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def intersected_circles(self, center, balls: list[Ball]) -> list[Ball]:
        intersections = []
        for ball in balls:
            if (np.linalg.norm(np.array(center) - np.array(ball.center))) < Ball.r * 2:
                intersections.append(ball)
        return intersections

    def find_balls(self, img_bgr: cv2.Mat, table_mask=None):

        # find circles using contours

        table_mask = (
            self.create_table_mask(img_bgr) if table_mask is None else table_mask
        )

        balls_mask = cv2.bitwise_not(
            table_mask[self.y : self.y + self.h, self.x : self.x + self.w]
        )

        contours, _ = cv2.findContours(
            balls_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = [c for c in contours if cv2.contourArea(c) > 1000]

        balls: list[Ball] = []

        for cnt in contours:
            (x, y), r = cv2.minEnclosingCircle(cnt)
            if r < 30 and r > 23:
                balls.append(Ball(x, y, r))

        # find circles using HoughCircles

        table = img_bgr[self.y : self.y + self.h, self.x : self.x + self.w]
        table_gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(
            table_gray,
            cv2.HOUGH_GRADIENT,
            1,
            10,
            param1=50,
            param2=30,
            minRadius=23,
            maxRadius=30,
        )

        if circles is None:
            return []

        for removed in circles[0, :]:
            c = (removed[0], removed[1])
            r = removed[2]
            if self.intersected_circles(c, balls) or r < 23 or r > 27:
                continue
            balls.append(Ball(removed[0], removed[1], removed[2]))

        ghost = self.find_ghost(img_bgr, balls)

        balls = [b for b in balls if b != ghost]
        Ball.r = sum([b.r for b in balls]) / len(balls)

        return balls

    def select_ball(self, img_bgr: cv2.Mat, balls: list[Ball], title: str):
        temp_img = img_bgr.copy()
        letters = list(range(ord("a"), ord("z") + 1))
        temp_letters = letters.copy()
        selected = None
        for ball in balls:
            if ball.type < 0:
                ball.key = temp_letters.pop(0)

        while selected is None:
            for ball in balls:

                cv2.circle(
                    temp_img,
                    (ball.intX + self.x, ball.intY + self.y),
                    ball.intR,
                    ball.color,
                    2,
                )

                if ball.key > 0:
                    cv2.putText(
                        temp_img,
                        ball.upperKeyChr,
                        (
                            ball.intX + self.x,
                            ball.intY + self.y + 50,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

                cv2.rectangle(
                    temp_img,
                    (self.x, self.y),
                    (self.right, self.bottom),
                    (255, 0, 255),
                    1,
                )
            title = "Select a ball" if title is None else title
            cv2.imshow(
                title,
                cv2.cvtColor(
                    temp_img[
                        self.top - 30 : self.bottom + 30,
                        self.left - 30 : self.right + 30,
                    ],
                    cv2.COLOR_BGR2RGB,
                ),
            )
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()

            if key in [b.key for b in balls]:
                ball = [ball for ball in balls if ball.key == key][0]
                selected = ball if ball.type < 0 else None
            elif key == 27:
                return None
            else:
                continue

        return selected

    def select_ball_to_hit(self, img_bgr: cv2.Mat, balls):
        return self.select_ball(img_bgr, balls, "Select target ball")

    def find_pockets(self, img_bgr: cv2.Mat, cue: Ball, target: Ball):
        ...

    def select_pocket(self, img_bgr):

        l = 0
        r = self.w
        t = Ball.r + 3
        b = self.h - Ball.r - 3
        pockets = {
            1: Ball(l + Ball.r, t),
            2: Ball(l + (r - l) // 2, t),
            3: Ball(r - Ball.r, t),
            4: Ball(l + Ball.r, b),
            5: Ball(l + (r - l) // 2, b),
            6: Ball(r - Ball.r, b),
        }

        for pocket in pockets.keys():
            p = pockets[pocket]
            alt = {
                1: [
                    Ball(p.x + 15, p.y),
                    Ball(p.x, p.y + 15),
                    Ball(p.x + 7.5, p.y + 7.5),
                ],
                2: [p, Ball(p.x - 25, p.y), Ball(p.x + 25, p.y)],
                3: [
                    Ball(p.x - 15, p.y),
                    Ball(p.x, p.y + 15),
                    Ball(p.x - 7.5, p.y + 7.5),
                ],
                4: [
                    Ball(p.x + 15, p.y),
                    Ball(p.x, p.y - 15),
                    Ball(p.x + 7.5, p.y - 7.5),
                ],
                5: [p, Ball(p.x - 25, p.y), Ball(p.x + 25, p.y)],
                6: [
                    Ball(p.x - 15, p.y),
                    Ball(p.x, p.y - 15),
                    Ball(p.x - 7.5, p.y - 7.5),
                ],
            }
            pockets[pocket].alt = alt[pocket]

        return self.select_ball(img_bgr, pockets.values(), "Select pocket")

    def calc(self, cue: Ball, target: Ball, pocket: Ball, img):
        pocket.alt.sort(key=lambda p: target.distance(p))
        pocket = pocket.alt[0]

        pocket_distance = target.distance(pocket)

        rx = (target.x - pocket.x) / pocket_distance
        ry = (target.y - pocket.y) / pocket_distance
        cue_hit_point = (
            target.x + rx * (Ball.r * 2),
            target.y + ry * (Ball.r * 2),
        )
        hit_ball = Ball(cue_hit_point[0], cue_hit_point[1])

        cv2.line(
            img,
            (int(self.x + target.x), int(self.y + target.y)),
            (int(self.x + pocket.x), int(self.y + pocket.y)),
            (0, 0, 255),
            2,
        )
        cv2.line(
            img,
            (int(self.x + hit_ball.x), int(self.y + hit_ball.y)),
            (int(self.x + cue.x), int(self.y + cue.y)),
            (0, 0, 255),
            2,
        )
        key = self.draw_balls(img, [cue, target, pocket, hit_ball])
        if key == 27:
            return None

        return cue_hit_point

    def extend(self, cue: Ball, hit_point):
        dx = hit_point[0] - cue.x
        dy = hit_point[1] - cue.y
        ratio_xy = dx / dy

        ex = 0
        ey = 0

        # towards right
        if dx > 0:
            ex = self.w - cue.x
        # towards left
        else:
            ex = -cue.x
        # towards down
        if dy > 0:
            ey = self.h - cue.y
        # towards up
        else:
            ey = -cue.y

        if (ex / ey) / ratio_xy > 1:
            ex = ey * ratio_xy
        else:
            ey = ex / ratio_xy

        return (cue.x + ex, cue.y + ey)

    def draw_balls(self, img_bgr, balls: list[Ball]):
        img_bgr = img_bgr.copy()
        for ball in balls:
            cv2.circle(
                img_bgr,
                (ball.intX + self.x, ball.intY + self.y),
                ball.intR,
                ball.color,
                2,
            )
        cv2.imshow(
            "Balls",
            cv2.cvtColor(
                img_bgr[
                    self.top - 30 : self.bottom + 30, self.left - 30 : self.right + 30
                ],
                cv2.COLOR_BGR2RGB,
            ),
        )
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        return key

    def draw_contours(self, img_bgr, contours):
        img_bgr = img_bgr.copy()
        cv2.drawContours(img_bgr, contours, -1, (0, 255, 0), 3)
        cv2.imshow("Contours", cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_reachable_balls(self, img, cue: Ball):
        letters = list(range(ord("a"), ord("z") + 1))
        temp_letters = letters.copy()
        for ball in self.balls:
            if ball.type < 0:
                ball.key = temp_letters.pop(0)
                cv2.putText(
                    img,
                    ball.upperKeyChr,
                    (
                        ball.intX + self.x,
                        ball.intY + self.y + 50,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

        reachable_balls = cue.find_reachable_balls(self.balls)

        # draw reachable balls
        for reachable_ball in reachable_balls:
            r = reachable_ball
            random_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            length = 300000
            ball: Ball = reachable_ball[0]
            min_angle = reachable_ball[1]
            max_angle = reachable_ball[2]
            distance = reachable_ball[3]
            pt_ball = (cue.intX + self.x, cue.intY + self.y)
            pt_min_x = int(cue.x + math.cos(min_angle) * length) + self.x
            pt_min_y = int(cue.y + math.sin(min_angle) * length) + self.y
            pt_max_x = int(ball.x + math.cos(max_angle) * length) + self.x
            pt_max_y = int(ball.y + math.sin(max_angle) * length) + self.y
            pt_min = (pt_min_x, pt_min_y)
            pt_max = (pt_max_x, pt_max_y)

            thickness = 1

            cv2.line(img, pt_ball, pt_min, random_color, thickness)
            cv2.line(img, pt_ball, pt_max, random_color, thickness)

            cv2.putText(
                img,
                ball.upperKeyChr,
                (
                    ball.intX + self.x,
                    ball.intY + self.y + 50,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                random_color,
                2,
            )

    def run(self):
        img = self.screenshot()
        table_mask = self.create_table_mask(img)
        self.find_table_area(img, table_mask)

        while True:
            while kb.read_key() != "ctrl":
                pass

            img = self.screenshot()

            self.balls = self.find_balls(img)
            self.balls.sort(key=lambda b: b.x)

            cue = self.find_cue_ball(img, self.balls)

            reachable_balls = cue.find_reachable_balls(self.balls)

            # self.draw_reachable_balls(img, cue)

            target = self.select_ball_to_hit(img, reachable_balls[:, 0])
            if target is None:
                continue

            pocket = self.select_pocket(img)

            if pocket is None:
                continue

            hit_point = self.calc(cue, target, pocket, img)

            if hit_point is None:
                continue

            hit_point = self.extend(cue, hit_point)

            sleep(0.2)
            auto.moveTo(self.x + hit_point[0], self.y + hit_point[1], 0.4)
            sleep(0.1)
            auto.click()
            sleep(0.2)
            auto.moveTo(2560 / 2, 1440 / 4)
            sleep(0.2)


# strengths
forces = {
    1: 14,
    2: 21,
    3: 26,
    4: 31,
    5: 34,
    6: 38,
    7: 41,
    8: 43,
    # one wall hit
    9: 51,
    10: 59,
    11: 67,
    12: 73,
}
