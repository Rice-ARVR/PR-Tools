import cv2
import asyncio
import websockets
import ctypes
import win32gui
import win32ui
import win32con
import numpy as np
import time

TARGET_FPS = 60
JPEG_QUALITY = 75
PORT = 8765

PW_RENDERFULLCONTENT = 0x00000002
FRAME_INTERVAL = 1.0 / TARGET_FPS


def find_dust_window():
    result = {}

    def callback(hwnd, _):
        title = win32gui.GetWindowText(hwnd)
        if 'SUITS_DUST' in title and '64-bit' in title:
            result['hwnd'] = hwnd
            result['title'] = title

    win32gui.EnumWindows(callback, None)

    if not result:
        raise Exception("DUST window not found. Make sure the simulation is running.")

    print(f"Found window: '{result['title']}'")
    hwnd = result['hwnd']

    if win32gui.IsIconic(hwnd):
        raise Exception("DUST window is minimized — restore it first")

    return hwnd


def capture_window(hwnd):
    """Capture window contents via PrintWindow + BitBlt.
    Works when the window is backgrounded or partially occluded."""
    try:
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
    except Exception:
        return None

    w = right - left
    h = bottom - top

    if w <= 0 or h <= 0:
        return None

    hwnd_dc = mfc_dc = save_dc = bitmap = old_bitmap = None
    try:
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc  = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        bitmap  = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, w, h)

        # Track the default bitmap so we can re-select it before deletion,
        # which is required to avoid GDI handle exhaustion over time.
        old_bitmap = save_dc.SelectObject(bitmap)

        # PW_RENDERFULLCONTENT composites DirectX surfaces into the DC,
        # so capture works even when the window isn't on screen.
        ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)

        bmp_data   = bitmap.GetBitmapBits(True)
        frame_bgra = np.frombuffer(bmp_data, dtype=np.uint8).reshape(h, w, 4)
        return cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

    except Exception as e:
        print(f"[capture] Error: {e}")
        return None

    finally:
        # Must deselect bitmap before deleting it, and release in reverse order.
        try:
            if old_bitmap and save_dc: save_dc.SelectObject(old_bitmap)
            if bitmap:                 win32gui.DeleteObject(bitmap.GetHandle())
            if save_dc:                save_dc.DeleteDC()
            if mfc_dc:                 mfc_dc.DeleteDC()
            if hwnd_dc:                win32gui.ReleaseDC(hwnd, hwnd_dc)
        except Exception:
            pass


async def stream_frames(websocket):
    print(f"Client connected: {websocket.remote_address}")
    hwnd = find_dust_window()

    try:
        while True:
            t0 = time.monotonic()

            frame = capture_window(hwnd)
            if frame is None:
                await asyncio.sleep(0.001)
                continue

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            await websocket.send(buffer.tobytes())

            elapsed   = time.monotonic() - t0
            sleep_for = max(0.0, FRAME_INTERVAL - elapsed)
            await asyncio.sleep(sleep_for)

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")


async def main():
    print(f"Starting stream server on ws://0.0.0.0:{PORT}")
    async with websockets.serve(stream_frames, "0.0.0.0", PORT):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())