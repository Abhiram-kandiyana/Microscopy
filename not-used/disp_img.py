# show image like matlab where you can see pixel values, zoom etc
# usage: disp_img(window name, image)

from matplotlib import pyplot as plt
import cv2


def disp_img(win_name, img):
    # Attention: OpenCV uses BGR color ordering per default whereas
    # Matplotlib assumes RGB color ordering!
    plt.figure(win_name)
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    #plt.title()
    #plt.ion()
    print('Close the figure to continue.')
    plt.show()
    #plt.pause(0.001)
    #plt.show(block=False)
    #plt.hold(b=True)
    #cv2.waitKey(1000000)
