"""Temporary, small input masks for repeatedly detected stationary objects."""

from collections import deque
from dataclasses import dataclass, field
import math
from statistics import median
import time


@dataclass
class _Candidate:
    samples: deque = field(default_factory=lambda: deque(maxlen=256))


@dataclass
class _Mask:
    box: tuple
    expires_at: float
    first_seen: float
    misses: int = 0


class StaticObjectMasker:
    """Call apply before YOLO, then observe once with that inference's xyxy boxes.

    observe returns indices to retain in the YOLO result. This also suppresses
    the remembered stationary box if a small black patch fails to fool YOLO.
    Confidence changes do not reset masks. Only the model input is modified;
    retain the original separately, and annotate the model input to show masks.

    Like any stationary-object heuristic, this can also mask a motionless
    person. Masks are small, expire, and are periodically checked without patches.
    """

    def __init__(
        self,
        required_hits=4,          # Liczba zgodnych wykryć tworzących maskę; minimum 3.
        window_seconds=20,        # Sekundy historii używanej do uczenia nieruchomości.
        mask_seconds=120,         # Sekundy od utworzenia maski do jej wygaśnięcia.
        patch_size=15,            # Bok łaty w px oryginału; ograniczony rozmiarem obiektu.
        center_tolerance=12,      # Dopuszczalne różnice/rozrzut środka ramki w px.
        size_tolerance=0.35,      # Względna zmiana wymiarów ramki; minimum tolerancji to 4 px.
        movement_threshold=4,    # Przesunięcie w px przy spójnym ruchu (>= 80% drogi).
        probe_interval=15,       # Odstęp w sekundach między kontrolami bez maskowania.
        missing_probes=2,        # Liczba kolejnych kontroli bez obiektu usuwająca maskę.
        mask_iou_threshold=0.45,  # Minimalne IoU już zamrożonego obiektu (0..1); mniej = luźniej.
        log_events=True,         # Wypisz komunikat przy utworzeniu każdej nowej maski.
        min_size_tolerance=4,    # Minimalna tolerancja wymiarów ramki w px.
        directionality=0.8,      # Przesunięcie / cała droga, od którego ruch jest spójny.
        history_limit=256,       # Maksymalna liczba obserwacji jednego kandydata.
    ):
        if required_hits < 3 or patch_size < 1 or int(patch_size) != patch_size:
            raise ValueError("Use at least three hits and an integer patch size >= 1")
        if min(window_seconds, mask_seconds, center_tolerance,
               movement_threshold, probe_interval) <= 0:
            raise ValueError("Durations and distance thresholds must be positive")
        if not 0 < size_tolerance < 1 or missing_probes < 1:
            raise ValueError("Invalid size tolerance or missing-probe count")
        if not 0 < mask_iou_threshold <= 1:
            raise ValueError("Mask IoU threshold must be in (0, 1]")
        if min_size_tolerance < 0 or not 0 < directionality <= 1 or history_limit < required_hits:
            raise ValueError("Invalid dimension tolerance, directionality or history limit")
        self.required_hits = required_hits
        self.window_seconds = window_seconds
        self.mask_seconds = mask_seconds
        self.patch_size = int(patch_size)
        self.center_tolerance = center_tolerance
        self.size_tolerance = size_tolerance
        self.movement_threshold = movement_threshold
        self.probe_interval = probe_interval
        self.missing_probes = missing_probes
        self.mask_iou_threshold = mask_iou_threshold
        self.log_events = log_events
        self.min_size_tolerance = min_size_tolerance
        self.directionality = directionality
        self.history_limit = history_limit
        self._candidates = []
        self._masks = []
        self._last_probe = -math.inf
        self._probe_frame = False
        self._shape = None

    @property
    def active_boxes(self):
        """Frozen mask boxes as of the latest apply/observe call, for diagnostics."""
        return tuple(mask.box for mask in self._masks)

    @staticmethod
    def _center(box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    @staticmethod
    def _median_box(samples):
        return tuple(median(box[i] for _, box in samples) for i in range(4))

    def _same_object(self, first, second):
        a, b = self._center(first), self._center(second)
        if max(abs(a[0] - b[0]), abs(a[1] - b[1])) > self.center_tolerance:
            return False
        for lo, hi in ((0, 2), (1, 3)):
            size = first[hi] - first[lo]
            if abs((second[hi] - second[lo]) - size) > max(self.min_size_tolerance, size * self.size_tolerance):
                return False
        return True

    @staticmethod
    def _overlaps(first, second):
        return (min(first[2], second[2]) > max(first[0], second[0])
                and min(first[3], second[3]) > max(first[1], second[1]))

    def _matches_mask(self, reference, box):
        """Match a frozen object even if its patch makes YOLO shorten the box.

        Keep the reference frozen: repeatedly following shifted boxes could
        gradually grow/move an ignored area onto a different object.
        This relaxed rule is not used to learn stationary candidates.
        """
        if self._same_object(reference, box):
            return True
        width = max(0.0, min(reference[2], box[2]) - max(reference[0], box[0]))
        height = max(0.0, min(reference[3], box[3]) - max(reference[1], box[1]))
        intersection = width * height
        area_reference = (reference[2] - reference[0]) * (reference[3] - reference[1])
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        union = area_reference + area_box - intersection
        return union > 0 and intersection / union >= self.mask_iou_threshold

    def filter_alarm_records(self, records):
        """Remove static evidence retrospectively after observe has classified it.

        Records hold boxes, their optional box_confidences, and observed_at in
        monotonic seconds. A frame with another valid object still counts once.
        Call after every observe, including when no accepted detection remains.
        Persist the returned records: removed static boxes must not reappear
        when a mask expires. This method performs no file or network operations.
        """
        kept, discarded = [], []
        for record in records:
            indices = [
                i for i, box in enumerate(record['boxes'])
                if not any(mask.first_seen <= record['observed_at'] < mask.expires_at
                           and self._matches_mask(mask.box, box) for mask in self._masks)
            ]
            if not indices:
                discarded.append(record)
                continue
            remaining = dict(record)
            remaining['boxes'] = [record['boxes'][i] for i in indices]
            if 'box_confidences' in record:
                remaining['box_confidences'] = [record['box_confidences'][i] for i in indices]
                remaining['confidence'] = max(remaining['box_confidences'])
            kept.append(remaining)
        return kept, discarded

    def _expire(self, now):
        self._masks = [mask for mask in self._masks if now < mask.expires_at]
        for candidate in self._candidates:
            while candidate.samples and now - candidate.samples[0][0] > self.window_seconds:
                candidate.samples.popleft()
        self._candidates = [c for c in self._candidates if c.samples]

    def apply(self, frame, *, now=None):
        """Return model input. Never draw on the supplied original frame."""
        now = time.monotonic() if now is None else now
        shape = frame.shape[:2]
        if self._shape is not None and self._shape != shape:
            self._candidates.clear()
            self._masks.clear()
        self._shape = shape
        self._expire(now)
        self._probe_frame = not self._masks or now - self._last_probe >= self.probe_interval
        if self._probe_frame:
            self._last_probe = now
            return frame

        masked = frame.copy()
        height, width = shape
        for mask in self._masks:
            cx, cy = self._center(mask.box)
            # Keep the patch inside both the detected object and the image.
            size = max(1, min(self.patch_size, int(mask.box[2] - mask.box[0]),
                              int(mask.box[3] - mask.box[1])))
            left = int(round(cx)) - size // 2
            top = int(round(cy)) - size // 2
            x1, x2 = max(0, left), min(width, left + size)
            y1, y2 = max(0, top), min(height, top + size)
            if x2 > x1 and y2 > y1:
                masked[y1:y2, x1:x2] = 0
        return masked

    def _stationary(self, samples):
        if len(samples) < self.required_hits:
            return False
        reference = self._median_box(samples)
        if not all(self._same_object(reference, box) for _, box in samples):
            return False
        centers = [self._center(box) for _, box in samples]
        # Bound the whole trajectory, not just adjacent frames: otherwise a
        # slowly walking person could be learned as a stationary object.
        if any(max(p[i] for p in centers) - min(p[i] for p in centers)
               > self.center_tolerance for i in (0, 1)):
            return False
        displacement = math.dist(centers[0], centers[-1])
        path = sum(math.dist(a, b) for a, b in zip(centers, centers[1:]))
        if displacement >= self.movement_threshold and path and displacement / path >= self.directionality:
            return False
        return True

    def observe(self, boxes, *, now=None):
        """Learn from one inference and return allowed indices in input order."""
        now = time.monotonic() if now is None else now
        self._expire(now)
        valid = {}
        for index, coords in enumerate(boxes):
            box = tuple(float(value) for value in coords)
            if (len(box) == 4 and all(math.isfinite(value) for value in box)
                    and box[2] > box[0] and box[3] > box[1]):
                valid[index] = box

        if self._probe_frame:
            retained = []
            for mask in self._masks:
                if any(self._matches_mask(mask.box, box) for box in valid.values()):
                    mask.misses = 0
                elif any(self._overlaps(mask.box, box) for box in valid.values()):
                    # A substantially different box on an unmodified frame
                    # may be a person passing through this location.
                    continue
                else:
                    mask.misses += 1
                if mask.misses < self.missing_probes:
                    retained.append(mask)
            self._masks = retained

        # One-to-one nearest assignment prevents multiple boxes in one frame
        # from counting as multiple temporal observations of the same object.
        available = {
            index: box for index, box in valid.items()
            if not any(self._matches_mask(mask.box, box) for mask in self._masks)
        }
        pairs = []
        for ci, candidate in enumerate(self._candidates):
            reference = self._median_box(candidate.samples)
            for index, box in available.items():
                if self._same_object(reference, box):
                    pairs.append((math.dist(self._center(reference), self._center(box)), ci, index))
        assignments = {}
        used = set()
        for _, ci, index in sorted(pairs):
            if ci not in used and index not in assignments:
                assignments[index] = self._candidates[ci]
                used.add(ci)

        newly_masked = []
        for index, box in available.items():
            candidate = assignments.get(index)
            if candidate is None:
                candidate = _Candidate(samples=deque(maxlen=self.history_limit))
                self._candidates.append(candidate)
            if not candidate.samples or now > candidate.samples[-1][0]:
                candidate.samples.append((now, box))
            if self._stationary(list(candidate.samples)):
                frozen = self._median_box(candidate.samples)
                if not any(self._matches_mask(mask.box, frozen) for mask in self._masks):
                    self._masks.append(_Mask(frozen, now + self.mask_seconds,
                                             first_seen=candidate.samples[0][0]))
                    if self.log_events:
                        cx, cy = self._center(frozen)
                        duration = now - candidate.samples[0][0]
                        print(
                            f"[STATIC_MASK] Dodano maskę: środek=({cx:.1f}, {cy:.1f}), "
                            f"bbox={tuple(round(v, 1) for v in frozen)}, "
                            f"łata do {self.patch_size}x{self.patch_size} px, "
                            f"trafienia={len(candidate.samples)} w {duration:.1f}s "
                            f"(okno {self.window_seconds}s), czas={self.mask_seconds}s",
                            flush=True,
                        )
                newly_masked.append(candidate)
        self._candidates = [c for c in self._candidates if not any(c is n for n in newly_masked)]
        return [index for index, box in valid.items()
                if not any(self._matches_mask(mask.box, box) for mask in self._masks)]
