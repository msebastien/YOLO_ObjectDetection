import sdl2
import sdl2.ext
import cv2


class Window(object):
    def __init__(self, width, height, title="Video"):
        sdl2.ext.init()
        self.window = sdl2.ext.Window(title, size=(width, height))
        self.window.show()
        self.width, self.height = width, height

    def paint(self, img):
        img = cv2.resize(img, (self.width, self.height))

        # Retrieves a 3D numpy array that represents the pixel data of
        # the window's surface
        surface = sdl2.ext.pixels3d(self.window.get_surface())

        # Updates the pixel data of the window's surface with the resized image
        # img.swapaxes(0, 1) swaps the axes of the image array
        # to match the expected format of the SDL surface
        surface[:, :, 0:3] = img.swapaxes(0, 1)

        # Refreshes the window to display the updated surface
        self.window.refresh()

        return

    def close_requested(self):
        ret = False

        # Retrieves a list of SDL2 events currently in the event queue
        events = sdl2.ext.get_events()

        if sdl2.ext.quit_requested(events):
            ret = True
        elif sdl2.ext.key_pressed(events, sdl2.SDLK_q):
            ret = True

        return ret

    def close(self):
        self.window.close()
