import asyncio
import math
import threading

from reolink_aio.api import Host


class ReolinkController:
    def __init__(self, host, username, password, channel=0, *,
                 retry_seconds=5, watchdog_seconds=60,
                 light_seconds=30, siren_seconds=30, poll_seconds=0.5,
                 light_allowed=None, siren_allowed=None, autostart=True):
        self.host = host
        self.username = username
        self.password = password
        self.channel = channel
        self.retry_seconds = retry_seconds
        self.watchdog_seconds = watchdog_seconds
        self.light_seconds = light_seconds
        self.siren_seconds = siren_seconds
        self.poll_seconds = poll_seconds
        self.light_allowed = light_allowed or (lambda: True)
        self.siren_allowed = siren_allowed or (lambda: True)
        self.camera = None
        self.connected = False
        self.siren_active = False
        self.light_active = False
        self._connect_lock = asyncio.Lock()
        self.loop = asyncio.new_event_loop() if autostart else None
        if autostart:
            threading.Thread(target=self._run_loop, daemon=True).start()

    async def connect(self, allowed=None):
        async with self._connect_lock:
            while not self.connected:
                if allowed is not None and not allowed():
                    return False
                try:
                    self.camera = Host(host=self.host, username=self.username, password=self.password)
                    await self.camera.get_host_data()
                    await self.camera.get_states()
                    self.connected = True
                    print("CONNECTED REOLINK")
                except Exception as error:
                    self.connected = False
                    print(f"Reolink: połączenie nieudane ({type(error).__name__}).")
                    await asyncio.sleep(self.retry_seconds)
        return True

    async def watchdog(self):
        while True:
            try:
                await self.camera.get_host_data()
                await self.camera.get_states()
            except Exception:
                self.connected = False
                await self.connect()
            await asyncio.sleep(self.watchdog_seconds)

    async def start(self):
        await self.connect()
        asyncio.create_task(self.watchdog())

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.create_task(self.start())
        self.loop.run_forever()

    async def _command(self, method, *args, allowed=None):
        if allowed is not None and not allowed():
            return False
        if not self.connected and not await self.connect(allowed):
            return False
        # Po oczekiwaniu na połączenie godziny mogły się już skończyć.
        if allowed is not None and not allowed():
            return False
        try:
            await getattr(self.camera, method)(*args)
        except Exception:
            self.connected = False
            if not await self.connect(allowed):
                return False
            if allowed is not None and not allowed():
                return False
            await getattr(self.camera, method)(*args)
        return True

    async def light_on(self):
        return await self._command('set_spotlight', self.channel, True, allowed=self.light_allowed)

    async def light_off(self):
        # Wyłączenie jest dozwolone także PO końcu harmonogramu.
        return await self._command('set_spotlight', self.channel, False)

    async def siren_on(self, duration=None):
        duration = self.siren_seconds if duration is None else duration
        return await self._command('set_siren', self.channel, True, int(math.ceil(duration)),
                                   allowed=self.siren_allowed)

    async def siren_off(self):
        return await self._command('set_siren', self.channel, False)

    async def _run_output(self, kind, seconds, turn_on, turn_off, allowed):
        active = kind + '_active'
        if getattr(self, active) or not allowed():
            return
        setattr(self, active, True)
        started = False
        try:
            started = await turn_on()
            if not started:
                return
            deadline = asyncio.get_running_loop().time() + seconds
            while allowed():
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(self.poll_seconds, remaining))
        except Exception as error:
            print(f"Reolink: błąd wyjścia {kind} ({type(error).__name__}).")
        finally:
            try:
                if started:
                    await turn_off()
            except Exception as error:
                self.connected = False
                print(f"Reolink: nie udało się wyłączyć {kind} ({type(error).__name__}).")
            finally:
                setattr(self, active, False)

    async def flash(self, seconds=None):
        seconds = self.light_seconds if seconds is None else seconds
        await self._run_output('light', seconds, self.light_on, self.light_off, self.light_allowed)

    async def _sound_alarm(self, seconds=None):
        seconds = self.siren_seconds if seconds is None else seconds
        await self._run_output('siren', seconds, lambda: self.siren_on(seconds),
                               self.siren_off, self.siren_allowed)

    def alarm(self, seconds=None):
        return asyncio.run_coroutine_threadsafe(self._alarm(seconds), self.loop)

    async def _alarm(self, seconds=None):
        # Każde wyjście ma własny harmonogram, czas działania i blokadę powtórzeń.
        await asyncio.gather(self.flash(seconds), self._sound_alarm(seconds))
