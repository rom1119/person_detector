import threading
import queue
import time


class NotificationWorker:

    def __init__(self, send_function):

        self.queue = queue.Queue()

        self.send_function = send_function

        self.running = True

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

                data = self.queue.get(
                    timeout=1
                )

                self.send_function(data[0], data[1])

                self.queue.task_done()


            except queue.Empty:
                continue