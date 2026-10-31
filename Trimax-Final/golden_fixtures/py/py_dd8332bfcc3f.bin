def move_mouse_relative_to_window(self, window, x, y):
        """
        Move the mouse to a specific location relative to the top-left corner
        of a window.

        :param x: the target X coordinate on the screen in pixels.
        :param y: the target Y coordinate on the screen in pixels.
        """
        _libxdo.xdo_move_mouse_relative_to_window(
            self._xdo, ctypes.c_ulong(window), x, y)