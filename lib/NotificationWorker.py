import threading
import queue
import time


class NotificationWorker:

    def __init__(self, send_function, queue_timeout=1):

        self.queue = queue.Queue()

        self.send_function = send_function

        self.running = True
        self.queue_timeout = queue_timeout

        self.thread = threading.Thread(
            target=self.worker,
            daemon=True
        )

        self.thread.start()


    def add(self, data):

        self.queue.put(data)


    def worker(self):

        while self.running:

            try:
                data = self.queue.get(timeout=self.queue_timeout)
            except queue.Empty:
                continue
            try:
                self.send_function(data[0], data[1])
            except Exception as error:
                # Nie loguj URL-a żądania, który może zawierać token bota.
                print(f"Nie udało się wysłać powiadomienia ({type(error).__name__}).")
            finally:
                self.queue.task_done()
