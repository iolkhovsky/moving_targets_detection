import argparse
import cv2
from imutils.video import FPS
import numpy as np

from cropper import Cropper
from dense_tracker import DenseFlowTracker
from clusterizer import Clusterizer
from utils import *

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="video",
                        help="Source of video stream")
    parser.add_argument("--videofile", type=str, default="test.mp4",
                        help="Path to the videofile")
    parser.add_argument("--camid", type=int, default=0,
                        help="Id of webcamera")
    parser.add_argument("--roi_tracker", type=str, default="kcf", #default="kcf",
                        help="Type of ROI tracker")
    parser.add_argument("--crop_x", type=int, default=960,
                        help="Crop size X (horizontal)")
    parser.add_argument("--crop_y", type=int, default=640,
                        help="Crop size Y (vertical)")
    return parser.parse_args()


def run(args):
    cap = None
    if args.source == "video":
        cap = cv2.VideoCapture(args.videofile)
    elif args.source == "camera":
        cap = cv2.VideoCapture(args.camid)
    else:
        raise ValueError("Invalid argument for video source")

    tracker = OPENCV_OBJECT_TRACKERS[args.roi_tracker]()
    roi_init_bbox = None
    fps = None
    frame = None
    visualization_frame = None
    box = None

    prev_roi_bbox = None
    prev_frame = None

    cropper = None
    if args.crop_x and args.crop_y:
        cropper = Cropper((None, None, args.crop_x, args.crop_y))

    dense_tracker = DenseFlowTracker()
    clusterizer = Clusterizer()

    while True:
        if frame is not None:
            prev_frame = frame.copy()
        ret, frame = cap.read()
        if not ret:
            break
        if cropper:
            frame = cropper(frame)
        visualization_frame = frame.copy()
        (H, W) = frame.shape[:2]

        if roi_init_bbox:
            prev_roi_bbox = box
            (success, box) = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                print("Roi: ", x, y, w, h)
                cv2.rectangle(visualization_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                common_roi = find_common_roi(box, prev_roi_bbox)
                if common_roi is not None:
                    if frame is not None and prev_frame is not None:
                        x, y, w, h = common_roi
                        prev_subframe = prev_frame[y:y+h, x:x+w]
                        subframe = frame[y:y+h, x:x+w]
                        mag, ang = dense_tracker.process(prev_subframe, subframe)
                        norm_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                        hsv = np.zeros_like(subframe)
                        hsv[..., 0] = ang * 180 / np.pi / 2
                        hsv[..., 1] = 255
                        hsv[..., 2] = norm_mag
                        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                        cv2.imshow('Flow', rgb)

                        median_speed = np.median(norm_mag)
                        ret, thresh1 = cv2.threshold(norm_mag, median_speed, 255, cv2.THRESH_BINARY)
                        at = cv2.adaptiveThreshold(norm_mag.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 9, -5)
                        cv2.imshow('thresh', thresh1)
                        cv2.imshow('ad_thresh', at)

                        '''Z = rgb.reshape((-1, 3))
                        # convert to np.float32
                        Z = np.float32(Z)
                        # define criteria, number of clusters(K) and apply kmeans()
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                        K = int(rgb.shape[0] * rgb.shape[1] / 1200.) * 2
                        if K > 4:
                            K = 4
                        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                        # Now convert back into uint8, and make original image
                        center = np.uint8(center)
                        res = center[label.flatten()]
                        res2 = res.reshape((rgb.shape))
                        cv2.imshow('res2', res2)'''

                        #clusters, colormap = clusterizer(mag, ang)
                        #cv2.imshow('Clusters', colormap)
                        #ret, thresh1 = cv2.threshold(mag, 0.75 * np.max(mag), np.max(mag), cv2.THRESH_BINARY)
                        #norm_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        #ret, thresh = cv2.threshold(norm_mag, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                        #th3 = cv2.adaptiveThreshold(norm_mag, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                        #                            cv2.THRESH_BINARY, 9, -5)
                        #cv2.imshow('Motion', thresh)
            fps.update()
            fps.stop()

            info = [
                ("Tracker", args.roi_tracker),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
            ]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(visualization_frame, text, (10, 20 + ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Stream", visualization_frame)

        key = cv2.waitKey(33) & 0xFF
        if key == ord("t"):
            roi_init_bbox = cv2.selectROI("Stream", visualization_frame, fromCenter=False, showCrosshair=True)
            tracker.init(frame, roi_init_bbox)
            fps = FPS().start()
        elif key == ord("q"):
            break
        elif key == ord("p"):
            pause_stop = False
            while not pause_stop:
                pause_stop = cv2.waitKey(10) & 0xFF == ord("p")

    cap.release()
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    cmdline_args = parse_cmd_args()
    run(cmdline_args)
