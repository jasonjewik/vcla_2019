# Done by Frannecklp

import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api

def grab_screen(region=None):

    hwin = win32gui.GetDesktopWindow()

    if region:
            left,top,x2,y2 = region
            width = x2 - left + 1
            height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


class Screen:
    def __init__(self,region=None):
        if region:
                self.left,self.top,x2,y2 = region
                self.width = x2 - self.left + 1
                self.height = y2 - self.top + 1
        else:
            self.width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
            self.height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
            self.left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
            self.top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

        self.hwin = win32gui.GetDesktopWindow()
        self.hwindc = win32gui.GetWindowDC(self.hwin)
        self.srcdc = win32ui.CreateDCFromHandle(self.hwindc)
        self.memdc = self.srcdc.CreateCompatibleDC()
        self.bmp = win32ui.CreateBitmap()
        self.bmp.CreateCompatibleBitmap(self.srcdc, self.width, self.height)


    def getWindow():
        self.memdc.SelectObject(self.bmp)
        self.memdc.BitBlt((0, 0), (self.width, self.height), self.srcdc, (self.left, self.top), win32con.SRCCOPY)
        signedIntsArray = self.bmp.GetBitmapBits(True)
        self.img = np.fromstring(signedIntsArray, dtype='uint8')
        self.img.shape = (height,width,4)
        return self.img

    def terminateProcess():
        self.srcdc.DeleteDC()
        self.memdc.DeleteDC()
        self.win32gui.ReleaseDC(self.hwin, hwindc)
        self.win32gui.DeleteObject(bmp.GetHandle())
