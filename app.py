#!/usr/bin/env python3
"""CHUNK 11: entry point. Starts receiver + processor threads, then the dashboard."""
import threading
from settings import CFG, RADIUS
from receiver import receiver_loop
from processor import processor_loop
from web import app

if __name__ == "__main__":
    threading.Thread(target=receiver_loop, daemon=True).start()
    threading.Thread(target=processor_loop, daemon=True).start()
    print("Smart Hearing Aid - Raspberry Pi 5")
    print(f"Dashboard: http://0.0.0.0:{CFG['web_port']}   array radius {RADIUS} m")
    app.run(host=CFG["web_host"], port=int(CFG["web_port"]), threaded=True)
