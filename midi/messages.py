from __future__ import annotations
import time
from typing import Optional, List
import mido

def list_output_ports() -> List[str]:
    return mido.get_output_names()

def find_output_port(name: Optional[str] = None, name_contains: Optional[str] = None) -> str:
    ports = list_output_ports()
    if not ports:
        raise ValueError("找不到任何 MIDI 輸出埠。請檢查是否已建立虛擬埠或接上裝置。")

    if name:
        for p in ports:
            if p == name:
                return p
        raise ValueError(f"找不到名為「{name}」的輸出埠。\n可用清單：{ports}")

    if name_contains:
        for p in ports:
            if name_contains.lower() in p.lower():
                return p
        raise ValueError(f"沒有包含「{name_contains}」關鍵字的輸出埠。\n可用清單：{ports}")

    return ports[0]

def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))

class MidiOut:
    def __init__(self, port_name: str):
        self.port_name = port_name
        self.port = mido.open_output(port_name)

    @classmethod
    def open(cls, name: Optional[str] = None, name_contains: Optional[str] = None) -> "MidiOut":
        port_name = find_output_port(name=name, name_contains=name_contains)
        return cls(port_name)

    @staticmethod
    def hz_to_midi(freq_hz: float) -> int:
        import math
        if freq_hz <= 0:
            return 0
        return int(round(69 + 12 * math.log2(freq_hz / 440.0)))

    def note_on(self, channel: int, note: int, velocity: int = 100):
        channel = clamp(channel, 0, 15)
        note = clamp(note, 0, 127)
        velocity = clamp(velocity, 0, 127)
        self.port.send(mido.Message('note_on', channel=channel, note=note, velocity=velocity))

    def note_off(self, channel: int, note: int, velocity: int = 64):
        channel = clamp(channel, 0, 15)
        note = clamp(note, 0, 127)
        velocity = clamp(velocity, 0, 127)
        self.port.send(mido.Message('note_off', channel=channel, note=note, velocity=velocity))

    def cc(self, channel: int, control: int, value: int):
        channel = clamp(channel, 0, 15)
        control = clamp(control, 0, 127)
        value = clamp(value, 0, 127)
        self.port.send(mido.Message('control_change', channel=channel, control=control, value=value))

    def program_change(self, channel: int, program: int):
        channel = clamp(channel, 0, 15)
        program = clamp(program, 0, 127)
        self.port.send(mido.Message('program_change', channel=channel, program=program))

    def pitch_bend(self, channel: int, value: int):
        channel = clamp(channel, 0, 15)
        value = max(-8192, min(8191, value))
        self.port.send(mido.Message('pitchwheel', channel=channel, pitch=value))

    def all_notes_off(self, channel: Optional[int] = None):
        if channel is None:
            for ch in range(16):
                self.cc(ch, 123, 0)
        else:
            self.cc(clamp(channel, 0, 15), 123, 0)

    def panic(self):
        for ch in range(16):
            self.cc(ch, 120, 0)
            self.cc(ch, 123, 0)

    def close(self):
        try:
            self.port.close()
        except Exception:
            pass

    def __enter__(self) -> "MidiOut":
        return self
    def __exit__(self, exc_type, exc, tb):
        self.close()

if __name__ == "__main__":
    print("Available ports", list_output_ports())
    with MidiOut.open(name="Virtual MIDI Port 1") as midi:
        midi.program_change(0, 0)
        midi.cc(0, 1, 0)
        midi.note_on(0, 60, 100)
        for i in range(100):
            time.sleep(0.05)
            midi.cc(0, 1, i+1)
        midi.note_off(0, 60, 64)
