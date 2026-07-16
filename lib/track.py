from collections import deque
import time


class Track:

    def __init__(self, center):
        self.center = center
        self.last_seen = time.time()
        self.ignore_until = 0

        self.history = deque()
        
    def __repr__(self):
        return (
            f"Track("
            f"center={self.center}, "
            f"last_seen={self.last_seen:.2f}, "
            f"ignore_until={self.ignore_until:.2f}, "
            f"history={len(self.history)})"
        )

    def add_detection(self, center):

        now = time.time()

        self.center = center
        self.last_seen = now

        self.history.append(
            (now, center[0], center[1])
        )

    def cleanup(self, max_age):

        now = time.time()

        while self.history:

            if now - self.history[0][0] > max_age:
                self.history.popleft()
            else:
                break
