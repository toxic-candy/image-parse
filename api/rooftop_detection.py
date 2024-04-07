from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math 
from shapely.geometry import Polygon
import math
from typing import TypedDict

zoom = 20
tileSize = 256
initialResolution = 2 * math.pi * 6378137 / tileSize
originShift = 2 * math.pi * 6378137 / 2.0
earthc = 6378137 * 2 * math.pi
factor = math.pow(2, zoom)
map_width = 256 * (2 ** zoom)

# pl = No of panels together as length commonside, pw = Same as for pw here w = width
# l = Length of panel in mm, w = Width of panel in mm
# solar_angle = Angle for rotation
pl, pw, l, w, solar_angle = 4, 1, 5, 5, 30

def to_gray_colorspace(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
def white_img_of_same_shape(im):
    return cv2.bitwise_not(np.zeros(im.shape, np.uint8))
def pixels_per_mm(lat, length):
    return length / math.cos(lat * math.pi / 180) * earthc * 1000 / map_width
def alt_sharp(gray):
    blur = cv2.bilateralFilter(gray, 5, sigmaColor=7, sigmaSpace=5)
    kernel_sharp = np.array((
        [-2, -2, -2],
        [-2, 17, -2],
        [-2, -2, -2]), dtype='int')
    return cv2.filter2D(blur, -1, kernel_sharp)

def sharp(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def contours_canny(cnts):
    cv2.drawContours(canny_contours, cnts, -1, 255, 1)

    # Removing the contours detected inside the roof
    for cnt in cnts:
        counters = 0
        cnt = np.array(cnt)
        cnt = np.reshape(cnt, (cnt.shape[0], cnt.shape[2]))
        pts = []

        if cv2.contourArea(cnt) > 10:
            for i in cnt:
                x, y = i
                if edged[y, x] == 255:
                    counters += 1
                    pts.append((x, y))

        if counters > 10:
            pts = np.array(pts)
            pts = pts.reshape(-1, 1, 2)
            cv2.polylines(canny_polygons, [pts], True, 0)


def contours_img(cnts):
    cv2.drawContours(image_contours, cnts, -1, 255, 1)

    # Removing the contours detected inside the roof
    for cnt in cnts:
        counter = 0
        cnt = np.array(cnt)
        cnt = np.reshape(cnt, (cnt.shape[0], cnt.shape[2]))
        pts = []
        if cv2.contourArea(cnt) > 5:
            for i in cnt:
                x, y = i
                if edged[y, x] == 255:
                    counter += 1
                    pts.append((x, y))
        if counter > 10:
            pts = np.array(pts)
            pts = pts.reshape(-1, 1, 2)
            cv2.polylines(image_polygons, [pts], True, 0)


def rotation(center_x, center_y, points, ang):
    angle = ang * math.pi / 180
    rotated_points = []
    for p in points:
        x, y = p
        x, y = x - center_x, y - center_y
        x, y = (x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle))
        x, y = x + center_x, y + center_y
        rotated_points.append((x, y))
    return rotated_points


def createLineIterator(P1, P2, img):
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:  # vertical line segment
        itbuffer[:, 0] = P1X
        if negY:
            itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
    elif P1Y == P2Y:  # horizontal line segment
        itbuffer[:, 1] = P1Y
        if negX:
            itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
        else:
            itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
    else:  # diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(float) / dY.astype(float)
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(int) + P1X
        else:
            slope = dY.astype(float) / dX.astype(float)
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(int) + P1Y

    # Remove points outside of image
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

    # Get intensities from img ndarray
    itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

    return itbuffer

class PanelData(TypedDict):
    image_with_panels: cv2.Mat
    area_of_panels: float
    num_panels: int

def panel_rotation(n_panels_in_series, solar_roof_img) -> PanelData:
    high_reso_solar = cv2.pyrUp(solar_roof_img)
    rows, cols = high_reso_solar.shape
    white_of_orig_img = white_img_of_same_shape(orig_image)
    high_reso_white_orig_img = cv2.pyrUp(white_of_orig_img)

    n_panels = 0

    for _ in range(1):
        for col in range(0, cols, l + 1):
            for row in range(0, rows, w + 1):

                # Rectangular Region of interest for solar panel area
                solar_patch = high_reso_solar[row:row + (w + 1) * pw + 1, col:col + ((l * pl) + 3)]
                r, c = solar_patch.shape

                # Rotation of rectangular patch according to the angle provided
                patch_points = np.array([[col, row], [c + col, row], [c + col, r + row], [col, r + row]], np.int32)
                rotated_patch_points = rotation((col + c) / 2, row + r / 2, patch_points, solar_angle)
                rotated_patch_points = np.array(rotated_patch_points, np.int32)

                # Check for if rotated points go outside of the image
                if (rotated_patch_points > 0).all():
                    solar_polygon = Polygon(rotated_patch_points)
                    polygon_points = np.array(solar_polygon.exterior.coords, np.int32)

                    # Appending points of the image inside the solar area to check the intensity
                    patch_intensity_check = []

                    # Point polygon test for each rotated solar patch area
                    for j in range(rows):
                        for k in range(cols):
                            if cv2.pointPolygonTest(polygon_points, (k, j), False) == 1:
                                patch_intensity_check.append(high_reso_solar[j, k])

                    # Check for the region available for Solar Panels
                    if np.mean(patch_intensity_check) == 255:

                        # Moving along the length of line to segment solar panels in the patch
                        solar_line_1 = createLineIterator(rotated_patch_points[0], rotated_patch_points[1], high_reso_solar)
                        solar_line_1 = solar_line_1.astype(int)
                        solar_line_2 = createLineIterator(rotated_patch_points[3], rotated_patch_points[2], high_reso_solar)
                        solar_line_2 = solar_line_2.astype(int)
                        line1_points = []
                        line2_points = []
                        if len(solar_line_2) > 10 and len(solar_line_1) > 10:

                            # Remove small unwanted patches
                            cv2.fillPoly(high_reso_solar, [rotated_patch_points], 0)
                            cv2.fillPoly(high_reso_white_orig_img, [rotated_patch_points], 0)
                            cv2.polylines(high_reso_orig, [rotated_patch_points], 1, 0, 2)
                            cv2.polylines(high_reso_white_orig_img, [rotated_patch_points], 1, 0, 2)

                            cv2.fillPoly(high_reso_orig, [rotated_patch_points], (255, 0, 0))
                            cv2.fillPoly(high_reso_white_orig_img, [rotated_patch_points], (255, 0, 0))

                            n_panels += 1
                            for i in range(5, len(solar_line_1), 5):
                                line1_points.append(solar_line_1[i])
                            for i in range(5, len(solar_line_2), 5):
                                line2_points.append(solar_line_2[i])

                        # Segmenting Solar Panels in the Solar Patch
                        for points1, points2 in zip(line1_points, line2_points):
                            x1, y1, _ = points1
                            x2, y2, _ = points2
                            cv2.line(high_reso_orig, (x1, y1), (x2, y2), (0, 0, 0), 1)
                            cv2.line(high_reso_white_orig_img, (x1, y1), (x2, y2), (0, 0, 0), 1)

        # Number of Solar Panels in series (3/4/5)
        n_panels_in_series = n_panels_in_series - 1

    BLUE_MIN = np.array([255, 0, 0], np.uint8) # pure blue
    BLUE_MAX = np.array([255, 50,50], np.uint8) # lighter blue
    dst = cv2.inRange(high_reso_orig, BLUE_MIN, BLUE_MAX)

    no_blue_pixels = cv2.countNonZero(dst)
    area_of_panels = no_blue_pixels*0.075

    return {"area_of_panels": area_of_panels, "image_with_panels": high_reso_orig, "num_panels": n_panels}

class RoofData():
    roof_area: float
    threshold_img: cv2.Mat

def get_roof_data(img: Image.Image, ) -> RoofData:

    # make sure the image is RGB
    img.convert('RGB')

    # Convert PIL image to OpenCV format
    global orig_image
    orig_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # latitude = ??
    # pl, pw, l, w, solar_angle = solar_panel_params()
    # length, width = pixels_per_mm(latitude)

    print(orig_image.shape)
    if orig_image.shape[0] > orig_image.shape[1]:
        pixels = min(orig_image.shape[0], 100)
        orig_image = cv2.resize(orig_image, (pixels, math.floor(orig_image.shape[0] / orig_image.shape[1] * pixels)))
    else:
        pixels = min(orig_image.shape[1], 100)
        orig_image = cv2.resize(orig_image, (math.floor(orig_image.shape[1] / orig_image.shape[0] * pixels), pixels))
    print(orig_image.shape)

    # Upscaling of Image
    global high_reso_orig
    high_reso_orig = cv2.pyrUp(orig_image)

    # White blank image for contours of Canny Edge Image
    global canny_contours
    canny_contours = white_img_of_same_shape(orig_image)

    # White blank image for contours of original image
    global image_contours
    image_contours = white_img_of_same_shape(orig_image)

    # White blank images removing rooftop's obstruction
    global image_polygons, canny_polygons
    image_polygons = to_gray_colorspace(canny_contours)
    canny_polygons = to_gray_colorspace(canny_contours)

    # Gray Image
    grayscale = to_gray_colorspace(orig_image)
    # plt.figure()
    # plt.title('grayscale')
    #plt.imshow(image, cmap='gray')

    # Edge Sharpened Image
    sharp_image = sharp(grayscale)
    # plt.figure()
    # plt.title('sharp_image')
    #plt.imshow(sharp_image, cmap='gray')

    # Canny Edge
    global edged
    edged = cv2.Canny(sharp_image, 180, 240)
    # plt.figure()
    # plt.title('edge_image')
    #plt.imshow(edged, cmap='gray')  

    # Otsu Threshold (Adaptive Threshold)
    # thresh = cv2.threshold(sharp_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.threshold(sharp_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # plt.figure()
    # plt.title('Threshold_image')
    #plt.imshow(thresh, cmap='gray')

    # Contours in Original Image
    contours_img(cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2])
    # Contours in Canny Edge Image
    contours_canny(cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2])

    # Optimum place for placing Solar Panels
    solar_roof = cv2.bitwise_and(image_polygons, canny_polygons)

    ret, thresh2 = cv2.threshold(sharp_image, 130, 255, cv2.THRESH_BINARY)
    n_white_pix = np.sum(thresh2==255)
    roof_area = n_white_pix*0.075
    print(roof_area)
    # plt.figure()
    # plt.title('threshold 2 based on sharp_image')
    # plt.imshow(thresh2, cmap='gray')
    # plt.show()

    # plt.axis('off')
    # plt.title("Roof with Panels (area = " + str(data["area_of_panels"])+')')
    # plt.imshow(data["image_with_panels"])

    return {"roof_area": roof_area, "threshold_img": thresh2}

def get_rainwater_data(image: Image.Image):
    roof_data = get_roof_data(image)
    return {"roof_area": roof_data["roof_area"], "threshold_img": roof_data["threshold_img"]}

def get_solar_data(image: Image.Image):
    roof_data = get_roof_data(image)
     # Rotation of Solar Panels
    import timeit
    st_time = timeit.default_timer()
    panel_data =  panel_rotation(pl, roof_data["threshold_img"])
    print(f"Time taken to process the image: {timeit.default_timer() - st_time} seconds")

    return {"area_of_panels": panel_data["area_of_panels"], "image_with_panels": panel_data["image_with_panels"]}