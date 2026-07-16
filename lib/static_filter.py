import math
import time

from .track import Track


class StaticObjectFilter:

    def __init__(
        self,
        static_time=30,
        ignore_time=120,
        match_distance=60,
        max_move=3
    ):

        self.static_time = static_time
        self.ignore_time = ignore_time
        self.match_distance = match_distance
        self.max_move = max_move

        self.tracks = []

    def should_ignore(self, center):

        track = self._get_track(center)

        track.add_detection(center)
        track.cleanup(self.static_time)
        print(f"added track WORKS !!! - {self.tracks}")
        now = time.time()

        if now < track.ignore_until:
            return True

        if self._is_static(track):

            print("STATIC OBJECT -> IGNORE")

            track.ignore_until = now + self.ignore_time

            return True

        return False

    def _get_track(self, center):

        now = time.time()

        self.tracks = [
            t for t in self.tracks
            if now - t.last_seen < self.ignore_time + 60
        ]

        best = None
        best_distance = 999999

        for track in self.tracks:

            dist = math.hypot(
                center[0] - track.center[0],
                center[1] - track.center[1]
            )

            if dist < self.match_distance and dist < best_distance:

                best = track
                best_distance = dist

        if best is not None:
            return best

        track = Track(center)

        self.tracks.append(track)

        return track

    def _is_static(self, track):

        history = track.history

        if len(history) < 4:
            return False

        if history[-1][0] - history[0][0] < self.static_time - 2:
            return False

        xs = [h[1] for h in history]
        ys = [h[2] for h in history]

        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)

        return (
            dx <= self.max_move
            and
            dy <= self.max_move
        )
