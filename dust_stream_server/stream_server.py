import cv2
import asyncio
import websockets
import ctypes
import win32gui
import win32ui
import win32con
import numpy as np
import time
from collections import defaultdict

TARGET_FPS = 60
JPEG_QUALITY = 75
PORT = 8765
CV_PORT = 8766          # separate port for your CV backend
PW_RENDERFULLCONTENT = 0x00000002
FRAME_INTERVAL = 1.0 / TARGET_FPS


# --- Shared broadcaster ---------------------------------------------------

class FrameBroadcaster:
    """Fans out encoded JPEG bytes to every registered async queue."""

    def __init__(self):
        self._queues: set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2)   # drop old frames if slow consumer
        async with self._lock:
            self._queues.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue):
        async with self._lock:
            self._queues.discard(q)

    async def publish(self, frame_bytes: bytes):
        async with self._lock:
            for q in self._queues:
                if q.full():
                    try:
                        q.get_nowait()   # evict stale frame so slow consumers don't block
                    except asyncio.QueueEmpty:
                        pass
                await q.put(frame_bytes)


broadcaster = FrameBroadcaster()


# --- Window capture (unchanged) ------------------------------------------

def find_dust_window():
    result = {}

    def callback(hwnd, _):
        title = win32gui.GetWindowText(hwnd)
        if 'SUITS_DUST' in title and '64-bit' in title:
            result['hwnd'] = hwnd
            result['title'] = title

    win32gui.EnumWindows(callback, None)
    if not result:
        raise Exception("DUST window not found.")
    print(f"Found window: '{result['title']}'")
    hwnd = result['hwnd']
    if win32gui.IsIconic(hwnd):
        raise Exception("DUST window is minimized — restore it first")
    return hwnd


def capture_window(hwnd):
    try:
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
    except Exception:
        return None
    w, h = right - left, bottom - top
    if w <= 0 or h <= 0:
        return None

    hwnd_dc = mfc_dc = save_dc = bitmap = old_bitmap = None
    try:
        hwnd_dc    = win32gui.GetWindowDC(hwnd)
        mfc_dc     = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc    = mfc_dc.CreateCompatibleDC()
        bitmap     = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
        old_bitmap = save_dc.SelectObject(bitmap)
        ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)
        bmp_data   = bitmap.GetBitmapBits(True)
        frame_bgra = np.frombuffer(bmp_data, dtype=np.uint8).reshape(h, w, 4)
        return cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
    except Exception as e:
        print(f"[capture] Error: {e}")
        return None
    finally:
        try:
            if old_bitmap and save_dc: save_dc.SelectObject(old_bitmap)
            if bitmap:                 win32gui.DeleteObject(bitmap.GetHandle())
            if save_dc:                save_dc.DeleteDC()
            if mfc_dc:                 mfc_dc.DeleteDC()
            if hwnd_dc:                win32gui.ReleaseDC(hwnd, hwnd_dc)
        except Exception:
            pass


# --- Capture loop (single producer) -------------------------------------

async def capture_loop():
    """Captures frames once and publishes encoded bytes to all subscribers."""
    hwnd = find_dust_window()
    while True:
        t0 = time.monotonic()
        frame = capture_window(hwnd)
        if frame is not None:
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            await broadcaster.publish(buf.tobytes())

        elapsed = time.monotonic() - t0
        await asyncio.sleep(max(0.0, FRAME_INTERVAL - elapsed))


# --- Client-facing websocket (port 8765) ---------------------------------

async def stream_to_client(websocket):
    print(f"Viewer connected: {websocket.remote_address}")
    q = await broadcaster.subscribe()
    try:
        while True:
            frame_bytes = await q.get()
            await websocket.send(frame_bytes)
    except websockets.exceptions.ConnectionClosed:
        print("Viewer disconnected")
    finally:
        await broadcaster.unsubscribe(q)


# --- CV-backend-facing websocket (port 8766) -----------------------------

async def stream_to_cv_backend(websocket):
    print(f"CV backend connected: {websocket.remote_address}")
    q = await broadcaster.subscribe()
    try:
        while True:
            frame_bytes = await q.get()
            await websocket.send(frame_bytes)
    except websockets.exceptions.ConnectionClosed:
        print("CV backend disconnected")
    finally:
        await broadcaster.unsubscribe(q)


# --- Entry point ---------------------------------------------------------

async def main():
    print(f"Stream server  → ws://0.0.0.0:{PORT}")
    print(f"CV feed server → ws://0.0.0.0:{CV_PORT}")

    async with (
        websockets.serve(stream_to_client,     "0.0.0.0", PORT),
        websockets.serve(stream_to_cv_backend, "0.0.0.0", CV_PORT),
    ):
        await capture_loop()


if __name__ == "__main__":
    asyncio.run(main())