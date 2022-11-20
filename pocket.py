from ball import Ball
from pool import Pool


class Pocket:
    pool: Pool = None
    _y_offset = 3

    alt = []

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    @classmethod
    def index(cls, index: int):
        l = 0
        r = Pocket.pool.w
        t = Ball.r + Pocket._y_offset
        b = Pocket.pool.h - Ball.r - Pocket._y_offset

        return [
            Pocket(l + Ball.r, t),
            Pocket(l + (r - l) // 2, t),
            Pocket(r - Ball.r, t),
            Pocket(l + Ball.r, b),
            Pocket(l + (r - l) // 2, b),
            Pocket(r - Ball.r, b),
        ][index]

    @classmethod
    def top_left(self):
        p = Pocket.index(0)
        p.alt = [
            Pocket(p.x + 15, p.y),
            Pocket(p.x, p.y + 15),
            Pocket(p.x + 7.5, p.y + 7.5),
        ]
        return p

    @classmethod
    def top_center(self):
        p = Pocket.index(1)
        p.alt = [p, Pocket(p.x - 25, p.y), Pocket(p.x + 25, p.y)]
        return p

    @classmethod
    def top_right(self):
        p = Pocket.index(2)
        p.alt = [
            Pocket(p.x - 15, p.y),
            Pocket(p.x, p.y + 15),
            Pocket(p.x - 7.5, p.y + 7.5),
        ]
        return p

    @classmethod
    def bottom_left(self):
        p = Pocket.index(3)
        p.alt = [
            Pocket(p.x + 15, p.y),
            Pocket(p.x, p.y - 15),
            Pocket(p.x + 7.5, p.y - 7.5),
        ]
        return p

    @classmethod
    def bottom_center(self):
        p = Pocket.index(4)
        p.alt = [p, Pocket(p.x - 25, p.y), Pocket(p.x + 25, p.y)]
        return p

    @classmethod
    def bottom_right(self):
        p = Pocket.index(5)
        p.alt = [
            Pocket(p.x - 15, p.y),
            Pocket(p.x, p.y - 15),
            Pocket(p.x - 7.5, p.y - 7.5),
        ]
        return p

    @property
    def center(self):
        return (self.x, self.y)
