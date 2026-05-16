# list_windows.py
import win32gui

def list_all_windows():
    windows = []
    win32gui.EnumWindows(lambda hwnd, _: windows.append(
        (hwnd, win32gui.GetWindowText(hwnd))
    ), None)
    for hwnd, title in windows:
        if title:
            print(f"'{title}'")

list_all_windows()