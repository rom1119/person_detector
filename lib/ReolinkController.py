import asyncio

from reolink_aio.api import Host
import threading


class ReolinkController:

    def __init__(
        self,
        host,
        username,
        password,
        channel=0
    ):

        self.host = host
        self.username = username
        self.password = password
        self.channel = channel

        self.camera = None
        self.connected = False

        self.loop = asyncio.new_event_loop()

        threading.Thread(
            target=self._run_loop,
            daemon=True
        ).start()

        self.siren_active = False
        self.light_active = False

    async def connect(self):

        while True:

            try:

                self.camera = Host(
                    host=self.host,
                    username=self.username,
                    password=self.password
                )


                await self.camera.get_host_data()

                await self.camera.get_states()

                self.connected = True

                print("CONNECTED REOLINK")

                return

            except Exception as e:

                print(e)

                self.connected = False

                await asyncio.sleep(5)
                
    async def watchdog(self):

        while True:

            try:
                await self.camera.get_host_data()
                await self.camera.get_states()

            except Exception:

                print("RECONNECT REOLINK")

                self.connected = False

                await self.connect()

            await asyncio.sleep(60)
            
    async def start(self):

        await self.connect()

        asyncio.create_task(
            self.watchdog()
        )

    def _run_loop(self):

        asyncio.set_event_loop(self.loop)

        self.loop.create_task(
            self.start()
        )

        self.loop.run_forever()



    async def light_on(self):
    
        if not self.connected:
            await self.connect()

        try:

            await self.camera.set_spotlight(
                self.channel,
                True
            )

        except Exception:

            self.connected = False
            await self.connect()



    async def light_off(self):

        if not self.connected:
            await self.connect()

        try:
            await self.camera.set_spotlight(
                self.channel,
                False
            )
        except Exception:

            self.connected = False
            await self.connect()

    async def siren_on(self, duration=5):

        if not self.connected:
            await self.connect()

        try:
            await self.camera.set_siren(
                self.channel,
                True,
                duration
            )
        except Exception:

            self.connected = False
            await self.connect()

    async def flash(self, seconds):

        if not self.connected:
            await self.connect()

        try:
            await self.light_on()

            await asyncio.sleep(seconds)

            await self.light_off()
        except Exception:

            self.connected = False
            await self.connect()

    def alarm(self, seconds = 10):

        asyncio.run_coroutine_threadsafe(
            self._alarm(seconds),
            self.loop
        )

    async def _alarm(self, seconds):

        if not self.connected:
            await self.connect()

        if self.siren_active:
            return

        try:
            self.siren_active = True

            await self.light_on()
            await self.siren_on(seconds)

            await asyncio.sleep(seconds)

            # await self.siren_off()
            await self.light_off()
            self.siren_active = False
        except Exception:
            self.siren_active = False
            self.connected = False
            await self.connect()
