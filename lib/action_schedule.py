"""Independent daily windows for detection, spotlight, Telegram and siren."""

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo


def parse_time(value, *, end=False):
    parts = str(value).split(':')
    try:
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) == 2 else 0
        if len(parts) > 2 or not 0 <= minute < 60 or not 0 <= hour <= 24:
            raise ValueError
        if hour == 24 and (minute != 0 or not end):
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError(f'Nieprawidłowa godzina {value!r}; użyj HH:MM (24:00 tylko jako koniec).') from None
    return hour * 60 + minute


@dataclass(frozen=True)
class DailyWindow:
    start: str = '00:00'
    end: str = '24:00'
    enabled: bool = True

    def __post_init__(self):
        parse_time(self.start)
        parse_time(self.end, end=True)

    def allows(self, at):
        if not self.enabled:
            return False
        start = parse_time(self.start)
        end = parse_time(self.end, end=True)
        minute = at.hour * 60 + at.minute
        if start == end:
            return True
        if start < end:
            return start <= minute < end
        return minute >= start or minute < end


class ActionPolicy:
    def __init__(self, timezone='Europe/Warsaw', *, detection=None, light=None,
                 telegram=None, siren=None, run_without_actions=False, clock=None):
        self.timezone = ZoneInfo(timezone)
        self.detection = detection or DailyWindow()
        self.windows = {name: window or DailyWindow()
                        for name, window in dict(light=light, telegram=telegram, siren=siren).items()}
        self.run_without_actions = run_without_actions
        self.clock = clock

    def now(self):
        value = self.clock() if self.clock else datetime.now(self.timezone)
        return self.local_time(value)

    def local_time(self, at):
        return (at.replace(tzinfo=self.timezone) if at.tzinfo is None
                else at.astimezone(self.timezone))

    def allowed(self, action, at=None):
        return self.windows[action].allows(self.now() if at is None else self.local_time(at))

    def should_detect(self, at=None):
        at = self.now() if at is None else self.local_time(at)
        return self.detection.allows(at) and (
            self.run_without_actions or any(window.allows(at) for window in self.windows.values()))
