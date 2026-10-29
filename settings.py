"""Konfiguracja PD_*; process environment > .env, nowe nazwy > stare aliasy."""
from dataclasses import dataclass
import math
import os
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
from lib.action_schedule import ActionPolicy, DailyWindow

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env', override=False)


@dataclass(frozen=True)
class EnvField:
    name: str
    default: str
    kind: str
    minimum: float | None
    maximum: float | None
    legacy: str | None
    category: str
    comment: str


FIELDS = (
    EnvField("TORCH_CPP_LOG_LEVEL", "ERROR", "str", None, None, None, "Model YOLO", "Poziom logów C++ PyTorch: INFO, WARNING, ERROR albo FATAL."),
    EnvField("JPEG_QUALITY", "95", "int", 0, 100, None, "Zdjęcia i katalogi", "Jakość zapisywanych zdjęć JPEG, 0..100."),
    EnvField("RTSP_URL", "", "str", None, None, "RTSP_URL", "Połączenia i dane dostępowe", "Adres strumienia RTSP (może zawierać hasło; nie publikuj .env)."),
    EnvField("REOLINK_HOST", "", "str", None, None, "REOLINK_IP", "Połączenia i dane dostępowe", "Adres kamery Reolink."),
    EnvField("REOLINK_USER", "", "str", None, None, "REOLINK_USER", "Połączenia i dane dostępowe", "Użytkownik kamery."),
    EnvField("REOLINK_PASSWORD", "", "str", None, None, "REOLINK_PASSWORD", "Połączenia i dane dostępowe", "Hasło kamery."),
    EnvField("TELEGRAM_TOKEN", "", "str", None, None, "TELEGRAM_TOKEN", "Połączenia i dane dostępowe", "Token bota Telegram."),
    EnvField("TELEGRAM_CHAT_ID", "", "str", None, None, "TELEGRAM_CHAT_ID", "Połączenia i dane dostępowe", "Identyfikator czatu Telegram."),
    EnvField("TIMEZONE", "Europe/Warsaw", "str", None, None, None, "Harmonogramy: czas lokalny, początek włącznie, koniec wyłącznie", "Strefa IANA; uwzględnia zmianę czasu."),
    EnvField("DETECTION_ENABLED", "true", "bool", None, None, None, "Harmonogramy: czas lokalny, początek włącznie, koniec wyłącznie", "Włącz przetwarzanie detekcji."),
    EnvField("DETECTION_START", "00:00", "str", None, None, None, "Harmonogramy: czas lokalny, początek włącznie, koniec wyłącznie", "Ogólne dopuszczalne godziny pracy detektora."),
    EnvField("DETECTION_END", "24:00", "str", None, None, None, "Harmonogramy: czas lokalny, początek włącznie, koniec wyłącznie", "Koniec ogólnego przedziału pracy detektora."),
    EnvField("RUN_WITHOUT_ACTIONS", "false", "bool", None, None, None, "Harmonogramy: czas lokalny, początek włącznie, koniec wyłącznie", "Detekcja/zapis także gdy żadna akcja nie jest obecnie dozwolona."),
    EnvField("LIGHT_ENABLED", "true", "bool", None, None, None, "Harmonogramy: czas lokalny, początek włącznie, koniec wyłącznie", "Światło po potwierdzonym wykryciu."),
    EnvField("LIGHT_START", "01:00", "str", None, None, None, "Harmonogramy: czas lokalny, początek włącznie, koniec wyłącznie", "Początek godzin reakcji światłem, np. 20:00."),
    EnvField("LIGHT_END", "24:00", "str", None, None, "HOUR_END", "Harmonogramy: czas lokalny, początek włącznie, koniec wyłącznie", "Koniec godzin światła; może być wcześniej niż początek (przez północ)."),
    EnvField("LIGHT_DURATION_SECONDS", "30", "float", 0.01, None, None, "Harmonogramy: czas lokalny, początek włącznie, koniec wyłącznie", "Czas świecenia po wykryciu; skracany po zamknięciu harmonogramu."),
    EnvField("TELEGRAM_ENABLED", "true", "bool", None, None, None, "Harmonogramy: czas lokalny, początek włącznie, koniec wyłącznie", "Włącz zdjęcia/powiadomienia."),
    EnvField("TELEGRAM_START", "01:00", "str", None, None, None, "Harmonogramy: czas lokalny, początek włącznie, koniec wyłącznie", "Początek godzin Telegrama."),
    EnvField("TELEGRAM_END", "24:00", "str", None, None, "HOUR_END", "Harmonogramy: czas lokalny, początek włącznie, koniec wyłącznie", "Koniec godzin Telegrama."),
    EnvField("SIREN_ENABLED", "true", "bool", None, None, None, "Harmonogramy: czas lokalny, początek włącznie, koniec wyłącznie", "Włącz syrenę kamery."),
    EnvField("SIREN_START", "01:00", "str", None, None, None, "Harmonogramy: czas lokalny, początek włącznie, koniec wyłącznie", "Początek godzin syreny."),
    EnvField("SIREN_END", "24:00", "str", None, None, "HOUR_END", "Harmonogramy: czas lokalny, początek włącznie, koniec wyłącznie", "Koniec godzin syreny."),
    EnvField("SIREN_DURATION_SECONDS", "30", "int", 1, None, None, "Harmonogramy: czas lokalny, początek włącznie, koniec wyłącznie", "Czas syreny po wykryciu; skracany po zamknięciu harmonogramu."),
    EnvField("ACTION_POLL_SECONDS", "0.5", "float", 0.01, None, None, "Harmonogramy: czas lokalny, początek włącznie, koniec wyłącznie", "Co ile sekund sprawdzać koniec harmonogramu aktywnego wyjścia."),
    EnvField("MODEL_PATH", "best.pt", "path", None, None, None, "Model YOLO", "Ścieżka wag; względna względem repozytorium."),
    EnvField("MODEL_DEVICE", "cpu", "str", None, None, None, "Model YOLO", "Urządzenie YOLO, np. cpu albo 0 dla GPU."),
    EnvField("MODEL_IMGSZ", "960", "int", 32, None, None, "Model YOLO", "Rozmiar wejścia modelu."),
    EnvField("MODEL_CONFIDENCE", "0.5", "float", 0, 1, None, "Model YOLO", "Minimalna pewność detekcji."),
    EnvField("MODEL_IOU", "0.5", "float", 0, 1, None, "Model YOLO", "Próg NMS YOLO; niezależny od MASK_IOU_THRESHOLD."),
    EnvField("MODEL_CLASSES", "0", "classes", None, None, None, "Model YOLO", "Klasy rozdzielone przecinkami; puste = wszystkie."),
    EnvField("MODEL_VERBOSE", "false", "bool", None, None, None, "Model YOLO", "Szczegółowe logi predykcji YOLO."),
    EnvField("LOOP_DELAY_SECONDS", "0.3", "float", 0, None, None, "Pętla i strumień RTSP", "Przerwa PO predykcji; dochodzi do czasu obliczeń CPU."),
    EnvField("IDLE_SLEEP_SECONDS", "60", "float", 0.01, None, None, "Pętla i strumień RTSP", "Przerwa poza godzinami działania."),
    EnvField("NO_FRAME_SLEEP_SECONDS", "0.8", "float", 0.01, None, None, "Pętla i strumień RTSP", "Przerwa, gdy kamera nie zwróciła klatki."),
    EnvField("RTSP_RECONNECT_SECONDS", "21600", "float", 0.01, None, None, "Pętla i strumień RTSP", "Okresowe odnowienie połączenia (21600 s = 6 h)."),
    EnvField("RTSP_MAX_FAILED_READS", "5", "int", 0, None, None, "Pętla i strumień RTSP", "Ponów połączenie po przekroczeniu tej liczby błędnych odczytów."),
    EnvField("RTSP_RETRY_SECONDS", "2", "float", 0.01, None, None, "Pętla i strumień RTSP", "Przerwa między próbami ponownego połączenia."),
    EnvField("RTSP_BUFFER_SIZE", "1", "int", 1, None, None, "Pętla i strumień RTSP", "Żądany bufor VideoCapture przy połączeniu."),
    EnvField("REOLINK_CHANNEL", "0", "int", 0, None, None, "Pętla i strumień RTSP", "Numer kanału kamery."),
    EnvField("REOLINK_RETRY_SECONDS", "5", "float", 0.01, None, None, "Pętla i strumień RTSP", "Odstęp ponawiania połączenia API kamery."),
    EnvField("REOLINK_WATCHDOG_SECONDS", "60", "float", 0.01, None, None, "Pętla i strumień RTSP", "Odstęp kontroli połączenia API kamery."),
    EnvField("ALARM_MIN_DETECTIONS", "7", "int", 1, None, None, "Potwierdzanie alarmu, analiza ruchu i stare filtrowanie statyczne", "Minimalna liczba kwalifikujących się klatek do reakcji."),
    EnvField("ALARM_WINDOW_SECONDS", "60", "float", 0.01, None, None, "Potwierdzanie alarmu, analiza ruchu i stare filtrowanie statyczne", "Okno klatek licznika alarmu."),
    EnvField("MOTION_HISTORY_SIZE", "5", "int", 2, None, None, "Potwierdzanie alarmu, analiza ruchu i stare filtrowanie statyczne", "Liczba ostatnich środków w dotychczasowym filtrze ruchu."),
    EnvField("MOTION_MIN_AVERAGE_PX", "3", "float", 0, None, None, "Potwierdzanie alarmu, analiza ruchu i stare filtrowanie statyczne", "Odrzuć ruch o mniejszej średniej odległości."),
    EnvField("MOTION_MAX_AVERAGE_PX", "480", "float", 0, None, None, "Potwierdzanie alarmu, analiza ruchu i stare filtrowanie statyczne", "Odrzuć ruch o większej średniej odległości."),
    EnvField("MOTION_MAX_JUMP_PX", "550", "float", 0, None, None, "Potwierdzanie alarmu, analiza ruchu i stare filtrowanie statyczne", "Maksymalny ostatni skok środka ramki."),
    EnvField("STATIC_SECONDS", "30", "float", 0.01, None, None, "Potwierdzanie alarmu, analiza ruchu i stare filtrowanie statyczne", "Wymagany czas nieruchomości starego filtra."),
    EnvField("STATIC_IGNORE_SECONDS", "120", "float", 0.01, None, None, "Potwierdzanie alarmu, analiza ruchu i stare filtrowanie statyczne", "Czas ignorowania przez stary filtr."),
    EnvField("STATIC_MATCH_DISTANCE_PX", "60", "float", 0.01, None, None, "Potwierdzanie alarmu, analiza ruchu i stare filtrowanie statyczne", "Odległość dopasowania środków w starym filtrze."),
    EnvField("STATIC_MAX_MOVE_PX", "3", "float", 0, None, None, "Potwierdzanie alarmu, analiza ruchu i stare filtrowanie statyczne", "Rozrzut środków uznawany za nieruchomy w starym filtrze."),
    EnvField("STATIC_MIN_OBSERVATIONS", "3", "int", 2, None, None, "Potwierdzanie alarmu, analiza ruchu i stare filtrowanie statyczne", "Minimalne obserwacje starego filtra."),
    EnvField("STATIC_TIME_TOLERANCE_SECONDS", "2", "float", 0, None, None, "Potwierdzanie alarmu, analiza ruchu i stare filtrowanie statyczne", "Tolerancja kompletności historii starego filtra."),
    EnvField("STATIC_RETENTION_EXTRA_SECONDS", "60", "float", 0, None, None, "Potwierdzanie alarmu, analiza ruchu i stare filtrowanie statyczne", "Dodatkowy czas pamiętania śladu przez stary filtr."),
    EnvField("PIXEL_HISTORY_SIZE", "10", "int", 1, None, None, "Jakość obrazu i odrzucanie uszkodzonych klatek", "Historia średnich zmian pikseli."),
    EnvField("ENTROPY_HISTORY_SIZE", "10", "int", 1, None, None, "Jakość obrazu i odrzucanie uszkodzonych klatek", "Historia entropii klatki."),
    EnvField("PIXEL_CHANGE_THRESHOLD", "1.6", "float", 0, None, None, "Jakość obrazu i odrzucanie uszkodzonych klatek", "Próg odchylenia średnich różnic pikseli od historii."),
    EnvField("ENTROPY_CHANGE_THRESHOLD", "1", "float", 0, None, None, "Jakość obrazu i odrzucanie uszkodzonych klatek", "Próg odchylenia entropii; obecnie służy do diagnostyki."),
    EnvField("DAMAGED_FRAME_SLEEP_SECONDS", "1", "float", 0, None, None, "Jakość obrazu i odrzucanie uszkodzonych klatek", "Przerwa po odrzuceniu uszkodzonej klatki."),
    EnvField("MASK_REQUIRED_HITS", "4", "int", 3, None, None, "Maskowanie statycznych fałszywych detekcji", "Zgodne obserwacje przed utworzeniem maski."),
    EnvField("MASK_WINDOW_SECONDS", "20", "float", 0.01, None, None, "Maskowanie statycznych fałszywych detekcji", "Okno zbierania obserwacji."),
    EnvField("MASK_DURATION_SECONDS", "120", "float", 0.01, None, None, "Maskowanie statycznych fałszywych detekcji", "Czas życia maski; kolejne wykrycia go nie przedłużają."),
    EnvField("MASK_PATCH_SIZE_PX", "15", "int", 1, None, None, "Maskowanie statycznych fałszywych detekcji", "Bok czarnego kwadratu w oryginalnej rozdzielczości."),
    EnvField("MASK_CENTER_TOLERANCE_PX", "12", "float", 0.01, None, None, "Maskowanie statycznych fałszywych detekcji", "Tolerancja środka i rozrzutu środków (na każdej osi)."),
    EnvField("MASK_SIZE_TOLERANCE", "0.35", "float", 0.00001, 0.99999, None, "Maskowanie statycznych fałszywych detekcji", "Względna tolerancja szerokości i wysokości (0.35 = 35%)."),
    EnvField("MASK_MOVEMENT_THRESHOLD_PX", "4", "float", 0.01, None, None, "Maskowanie statycznych fałszywych detekcji", "Spójne przesunięcie od tej wartości wyklucza nieruchomość."),
    EnvField("MASK_PROBE_INTERVAL_SECONDS", "15", "float", 0.01, None, None, "Maskowanie statycznych fałszywych detekcji", "Odstęp kontroli bez czarnych łat; bez dodatkowej predykcji."),
    EnvField("MASK_MISSING_PROBES", "2", "int", 1, None, None, "Maskowanie statycznych fałszywych detekcji", "Kolejne puste kontrole usuwające maskę."),
    EnvField("MASK_IOU_THRESHOLD", "0.45", "float", 0.00001, 1, None, "Maskowanie statycznych fałszywych detekcji", "Minimalne IoU już zapamiętanej ramki; mniej = luźniej."),
    EnvField("MASK_LOG_EVENTS", "true", "bool", None, None, None, "Maskowanie statycznych fałszywych detekcji", "Wypisz utworzenie maski."),
    EnvField("MASK_MIN_SIZE_TOLERANCE_PX", "4", "float", 0, None, None, "Maskowanie statycznych fałszywych detekcji", "Dolna granica tolerancji wymiarów ramki w px."),
    EnvField("MASK_DIRECTIONALITY", "0.8", "float", 0.00001, 1, None, "Maskowanie statycznych fałszywych detekcji", "Minimalny stosunek przesunięcia do całej drogi, aby uznać ruch za spójny."),
    EnvField("MASK_HISTORY_LIMIT", "256", "int", 3, None, None, "Maskowanie statycznych fałszywych detekcji", "Maksymalna liczba obserwacji jednego kandydata w pamięci."),
    EnvField("BUFFER_DIR", "files/buffer", "path", None, None, None, "Zdjęcia i katalogi", "Oryginały w buforze."),
    EnvField("BUFFER_ANNOTATED_DIR", "files/buff_annot", "path", None, None, None, "Zdjęcia i katalogi", "Maskowane predykcje YOLO w buforze."),
    EnvField("CLEAR_DIR", "files/clear", "path", None, None, None, "Zdjęcia i katalogi", "Docelowe oryginały po potwierdzeniu."),
    EnvField("ANNOTATED_DIR", "files/annotated", "path", None, None, None, "Zdjęcia i katalogi", "Docelowe predykcje z maską i ramkami; te zdjęcia idą na Telegram."),
    EnvField("DAMAGED_DIR", "files/damaged", "path", None, None, None, "Zdjęcia i katalogi", "Odrzucone klatki."),
    EnvField("ANNOTATION_LINE_WIDTH", "1", "int", 1, None, None, "Zdjęcia i katalogi", "Grubość ramki YOLO."),
    EnvField("ANNOTATION_FONT_SIZE", "1", "float", 0.01, None, None, "Zdjęcia i katalogi", "Rozmiar tekstu YOLO."),
    EnvField("CLEANUP_MAX_FILES", "1000", "int", 1, None, None, "Zdjęcia i katalogi", "Limit plików przy ręcznym/cronowym uruchomieniu cleanup.sh."),
    EnvField("DEBUG_LOGS", "true", "bool", None, None, None, "Zdjęcia i katalogi", "Diagnostyczne logi pętli, historii i bufora."),
    EnvField("TELEGRAM_SEND_FIRST_LAST_ONLY", "true", "bool", None, None, None, "Wysyłka Telegram i kolejka", "Wyślij pierwszą i ostatnią klatkę potwierdzonego zdarzenia; false = wszystkie."),
    EnvField("TELEGRAM_SEND_DELAY_SECONDS", "0.1", "float", 0, None, None, "Wysyłka Telegram i kolejka", "Przerwa przed pierwszą próbą wysłania zdjęcia."),
    EnvField("TELEGRAM_RETRY_SECONDS", "1", "float", 0, None, None, "Wysyłka Telegram i kolejka", "Przerwa przed ponowieniem wysyłki."),
    EnvField("TELEGRAM_ATTEMPTS", "2", "int", 1, None, None, "Wysyłka Telegram i kolejka", "Maksymalna liczba prób wysyłki."),
    EnvField("TELEGRAM_TIMEOUT_SECONDS", "15", "float", 0.01, None, None, "Wysyłka Telegram i kolejka", "Limit czasu pojedynczego żądania HTTP."),
    EnvField("TELEGRAM_MESSAGE_COOLDOWN_SECONDS", "2", "float", 0, None, None, "Wysyłka Telegram i kolejka", "Odstęp starej pomocniczej funkcji send_notification (nie bufora zdjęć)."),
    EnvField("QUEUE_TIMEOUT_SECONDS", "1", "float", 0.01, None, None, "Wysyłka Telegram i kolejka", "Oczekiwanie wątku kolejki na nowe zdjęcie."),
)


def _convert(field, raw):
    name = 'PD_' + field.name
    try:
        if field.kind == 'bool':
            values = {'true': True, '1': True, 'yes': True, 'on': True,
                      'false': False, '0': False, 'no': False, 'off': False}
            value = values[raw.strip().lower()]
        elif field.kind == 'int':
            value = int(raw)
        elif field.kind == 'float':
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError
        elif field.kind == 'classes':
            value = [int(part.strip()) for part in raw.split(',')] if raw.strip() else None
            if value is not None and any(item < 0 for item in value):
                raise ValueError
        elif field.kind == 'path':
            value = Path(raw).expanduser()
            if not value.is_absolute():
                value = BASE_DIR / value
        else:
            value = raw
        if field.minimum is not None and value < field.minimum:
            raise ValueError
        if field.maximum is not None and value > field.maximum:
            raise ValueError
    except (ValueError, TypeError, KeyError):
        # Never include the raw value: it could contain a secret.
        raise ValueError(f'Nieprawidłowa konfiguracja {name} (typ: {field.kind}).') from None
    return value


def load_settings(environ=None):
    environ = os.environ if environ is None else environ
    values = {}
    for field in FIELDS:
        raw = environ.get('PD_' + field.name)
        if raw is None and field.legacy:
            raw = environ.get(field.legacy)
        values[field.name] = _convert(field, field.default if raw is None else raw)
    cfg = SimpleNamespace(**values)
    if cfg.MOTION_MIN_AVERAGE_PX > cfg.MOTION_MAX_AVERAGE_PX:
        raise ValueError('PD_MOTION_MIN_AVERAGE_PX > PD_MOTION_MAX_AVERAGE_PX')
    if cfg.STATIC_TIME_TOLERANCE_SECONDS >= cfg.STATIC_SECONDS:
        raise ValueError('PD_STATIC_TIME_TOLERANCE_SECONDS musi być mniejsze od PD_STATIC_SECONDS')
    if cfg.MASK_HISTORY_LIMIT < cfg.MASK_REQUIRED_HITS:
        raise ValueError('PD_MASK_HISTORY_LIMIT musi pomieścić PD_MASK_REQUIRED_HITS')
    cfg.ACTION_POLICY = ActionPolicy(
        cfg.TIMEZONE,
        detection=DailyWindow(cfg.DETECTION_START, cfg.DETECTION_END, cfg.DETECTION_ENABLED),
        light=DailyWindow(cfg.LIGHT_START, cfg.LIGHT_END, cfg.LIGHT_ENABLED),
        telegram=DailyWindow(cfg.TELEGRAM_START, cfg.TELEGRAM_END, cfg.TELEGRAM_ENABLED),
        siren=DailyWindow(cfg.SIREN_START, cfg.SIREN_END, cfg.SIREN_ENABLED),
        run_without_actions=cfg.RUN_WITHOUT_ACTIONS,
    )
    cfg.MASK_OPTIONS = dict(
        required_hits=cfg.MASK_REQUIRED_HITS, window_seconds=cfg.MASK_WINDOW_SECONDS,
        mask_seconds=cfg.MASK_DURATION_SECONDS, patch_size=cfg.MASK_PATCH_SIZE_PX,
        center_tolerance=cfg.MASK_CENTER_TOLERANCE_PX, size_tolerance=cfg.MASK_SIZE_TOLERANCE,
        movement_threshold=cfg.MASK_MOVEMENT_THRESHOLD_PX,
        probe_interval=cfg.MASK_PROBE_INTERVAL_SECONDS, missing_probes=cfg.MASK_MISSING_PROBES,
        mask_iou_threshold=cfg.MASK_IOU_THRESHOLD, log_events=cfg.MASK_LOG_EVENTS,
        min_size_tolerance=cfg.MASK_MIN_SIZE_TOLERANCE_PX,
        directionality=cfg.MASK_DIRECTIONALITY, history_limit=cfg.MASK_HISTORY_LIMIT,
    )
    cfg.STATIC_OPTIONS = dict(
        static_time=cfg.STATIC_SECONDS, ignore_time=cfg.STATIC_IGNORE_SECONDS,
        match_distance=cfg.STATIC_MATCH_DISTANCE_PX, max_move=cfg.STATIC_MAX_MOVE_PX,
        min_observations=cfg.STATIC_MIN_OBSERVATIONS,
        time_tolerance=cfg.STATIC_TIME_TOLERANCE_SECONDS,
        retention_extra=cfg.STATIC_RETENTION_EXTRA_SECONDS,
        log_events=cfg.DEBUG_LOGS,
    )
    return cfg


def example_env():
    lines = ['# Skopiuj wybrane ustawienia do .env. Nie nadpisuj swoich danych dostępowych.',
             '# PD_* ma pierwszeństwo przed starymi nazwami. Zmiany wymagają restartu.',
             '# Godziny: HH:MM, przedziały przez północ działają; start == koniec = cała doba.',
             '# Każda akcja następuje dopiero po potwierdzonym wykryciu, nie od samej godziny.', '']
    # Grupuj także pola dodane później, bez powtarzania kategorii w szablonie.
    groups = {}
    for field in FIELDS:
        groups.setdefault(field.category, []).append(field)
    for category, fields in groups.items():
        lines.extend(['', '# --- ' + category + ' ---'])
        for field in fields:
            lines.extend(['# ' + field.comment, 'PD_' + field.name + '=' + field.default])
    return '\n'.join(lines) + '\n'


config = load_settings()
# Zachowane aliasy danych dostępowych dla dotychczasowych narzędzi.
RTSP_URL = config.RTSP_URL
REOLINK_IP = config.REOLINK_HOST
REOLINK_USER = config.REOLINK_USER
REOLINK_PASSWORD = config.REOLINK_PASSWORD
BOT_TOKEN = config.TELEGRAM_TOKEN
CHAT_ID = config.TELEGRAM_CHAT_ID
HOUR_END = int(os.environ.get('HOUR_END', '24'))
