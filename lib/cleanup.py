"""The existing buffer-count-based cleanup, with PD_* paths and limit."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import config


def cleanup(cfg=config):
    directories = (cfg.BUFFER_DIR, cfg.BUFFER_ANNOTATED_DIR, cfg.DAMAGED_DIR)
    files = sorted((p for p in directories[0].glob('*') if p.is_file()),
                   key=lambda path: path.stat().st_mtime)
    excess = max(0, len(files) - cfg.CLEANUP_MAX_FILES)
    # Zachowaj dotychczasową regułę: liczba usuwanych plików wynika z buffer.
    for directory in directories:
        oldest = sorted((p for p in directory.glob('*') if p.is_file()),
                        key=lambda path: path.stat().st_mtime)
        for path in oldest[:excess]:
            path.unlink()
    return excess


if __name__ == '__main__':
    print(f'Cleanup: usunięto do {cleanup()} najstarszych plików z każdego katalogu bufora.')
