"""Stimulator interfaces. program(condition_dict) blocks until the device is reprogrammed."""
from __future__ import annotations

import time


class DummyStimulator:
    """Stand-in: reprogramming takes reprog_ms."""
    def __init__(self, reprog_ms: float = 100.0, **_):
        self.reprog_s = float(reprog_ms) * 1e-3
        self.log = []

    def program(self, cond: dict):
        time.sleep(self.reprog_s); self.log.append(cond)

    def close(self):
        pass


class SerialStimulator:
    """Serial-port stimulator. command = Python format string over the condition's fields, e.g.
    "SET AMP {amplitude} POL {polarity} DUR {pulse_duration} NEG {contact_negative} POS {contact_positive}\\n".
    If `ack` is set, waits for a reply line starting with it (timeout = reprog_ms * 3)."""
    def __init__(self, port: str, command: str, baud: int = 115200, reprog_ms: float = 100.0, ack: str | None = None, **_):
        import serial                                  # pyserial
        self.reprog_s = float(reprog_ms) * 1e-3
        self.cmd, self.ack = command, ack
        self.ser = serial.Serial(port, baudrate=int(baud), timeout=self.reprog_s * 3)

    def program(self, cond: dict):
        self.ser.write(self.cmd.format(**cond).encode()); self.ser.flush()
        if self.ack:
            line = self.ser.readline().decode(errors="replace")
            if not line.startswith(self.ack):
                raise IOError(f"stimulator replied {line!r}")

    def close(self):
        self.ser.close()


STIMULATORS = {"dummy": DummyStimulator, "serial": SerialStimulator}
