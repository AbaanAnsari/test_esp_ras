"""CHUNK 6: UDP receive + packet validation + loss handling + 10 ms -> 20 ms assembly.
Does no DSP, so the socket is always drained quickly."""
import socket, struct, time, queue
import numpy as np
from settings import (UDP_HOST, UDP_PORT, RCVBUF, MAGIC, PACKET_VERSION, CH, BLOCK)
from state import state, state_lock, block_q, bump

HEADER = struct.Struct("<IIHBB")      # magic, seq, frames, channels, version  (12 bytes)
LATE_WINDOW = 100                     # seq this far behind = late/duplicate; further = ESP32 rebooted


def parse(data):
    """Return (seq, pcm[frames, CH] int16) or None if the packet is invalid."""
    if len(data) < HEADER.size:
        return None
    magic, seq, frames, channels, version = HEADER.unpack_from(data)
    if magic != MAGIC or version != PACKET_VERSION or channels != CH:
        return None
    if len(data) - HEADER.size != frames * channels * 2:
        return None
    pcm = np.frombuffer(data, dtype="<i2", offset=HEADER.size).reshape(frames, channels)
    return seq, pcm


def push_block(block):
    try:
        block_q.put_nowait(block)
    except queue.Full:
        try: block_q.get_nowait()
        except queue.Empty: pass
        block_q.put_nowait(block)
        bump("dropped_blocks")


def receiver_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RCVBUF)
    sock.bind((UDP_HOST, UDP_PORT))
    sock.settimeout(1.0)
    print(f"UDP receiver listening on {UDP_HOST}:{UDP_PORT}")

    expected, parts, have = None, [], 0

    while True:
        try:
            data, addr = sock.recvfrom(2048)
        except socket.timeout:
            expected, parts, have = None, [], 0      # stream went silent: resync on next packet
            with state_lock:
                if time.time() - state["last_update"] > 2.0:
                    state["status"] = "no packets from ESP32"
            continue
        except Exception as e:
            print("Receiver error:", e); time.sleep(0.1); continue

        p = parse(data)
        if p is None:
            continue
        seq, pcm = p

        if expected is not None:
            if seq < expected:
                if expected - seq <= LATE_WINDOW:
                    bump("late_packets"); continue          # late or duplicate: drop
                parts, have = [], 0                         # ESP32 rebooted: resync
            elif seq > expected:
                bump("packet_loss", seq - expected)
                parts, have = [], 0                         # gap: drop half-built block
        expected = (seq + 1) & 0xFFFFFFFF

        parts.append(pcm); have += pcm.shape[0]
        with state_lock:
            state["packets"] += 1
            state["last_sequence"] = int(seq)
            state["status"] = f"ESP32 {addr[0]}"
            state["last_update"] = time.time()

        if have >= BLOCK:
            blk = np.concatenate(parts)[:BLOCK].astype(np.float32) / 32768.0
            parts, have = [], 0
            push_block(np.ascontiguousarray(blk.T))        # [CH, BLOCK]
