import cv2
import numpy as np


def get8n(x, y, shape):
    out = []
    maxx = shape[0] - 1
    maxy = shape[1] - 1

    # top left
    outx = min(max(x - 1, 0), maxx)
    outy = min(max(y - 1, 0), maxy)
    out.append((outx, outy))

    # top center
    outx = x
    outy = min(max(y - 1, 0), maxy)
    out.append((outx, outy))

    # top right
    outx = min(max(x + 1, 0), maxx)
    outy = min(max(y - 1, 0), maxy)
    out.append((outx, outy))

    # left
    outx = min(max(x - 1, 0), maxx)
    outy = y
    out.append((outx, outy))

    # right
    outx = min(max(x + 1, 0), maxx)
    outy = y
    out.append((outx, outy))

    # bottom left
    outx = min(max(x - 1, 0), maxx)
    outy = min(max(y + 1, 0), maxy)
    out.append((outx, outy))

    # bottom center
    outx = x
    outy = min(max(y + 1, 0), maxy)
    out.append((outx, outy))

    # bottom right
    outx = min(max(x + 1, 0), maxx)
    outy = min(max(y + 1, 0), maxy)
    out.append((outx, outy))

    return out


def region_growing(img, seed,tres):
    # cv2.imshow("reg-grow input",img)
    # cv2.waitKey()
    seed_points = []
    outimg = np.zeros_like(img)
    seed_points.append((seed[0], seed[1]))
    processed = []
    seed_intensity = img[seed[0],seed[1]]
    # if(seed_intensity < tres):
    #     tres = img[seed[0],seed[1]]
    while (len(seed_points) > 0):
        pix = seed_points[0]
        outimg[pix[0], pix[1]] = 255
        for coord in get8n(pix[0], pix[1], img.shape):
            try:
                if img[coord[0], coord[1]] > tres:
                    outimg[coord[0], coord[1]] = 255
                    if not coord in processed:
                        seed_points.append(coord)
                    processed.append(coord)
            except:
                print("ignore this coordination point")
        seed_points.pop(0)
        # cv2.imshow("progress",outimg)
        # cv2.waitKey(1)
    return outimg


def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Seed: ' + str(x) + ', ' + str(y), img[y, x])
        clicks.append((y, x))

#both binary image input. Extracts seeds from seeds img and perform region growing for each
def apply_region_growing(image, seed,tres):
    # cv2.imshow("reg-grow-img",image);
    # cv2.waitKey()
    # _, _, _, seeds = cv2.connectedComponentsWithStats(seeds_img) # centroid of each blob as seed
    # seeds = seeds[1:]  # discard 0th being background
    result_mask = np.zeros_like(image)
    # for seed in seeds:
    out = region_growing(image, (int(seed[1]),int(seed[0])),tres)
    result_mask = cv2.bitwise_or(result_mask, out)
    return result_mask

if __name__ == "__main__":
    clicks = []
    image = cv2.imread(
        r'C:\Users\palakdave\MedicalImagingProject\Data\Leica_images\Send_to_Palak_2\Send_to_Palak\multi_in_single_out\exp\images1\PredictedMasks_png\pp_wo_mask_th\images1_Neo_cx_10_4_26.bmp',
        0)
    ret, img = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    cv2.namedWindow('Input')
    cv2.setMouseCallback('Input', on_mouse, 0, )
    cv2.imshow('Input', img)
    cv2.waitKey()
    seed = clicks[-1]
    out = region_growing(img, seed)
    cv2.imshow('Region Growing', out)
    cv2.waitKey()
    cv2.destroyAllWindows()