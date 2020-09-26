import argparse
import cv2
from imutils.video import FPS

from cropper import Cropper
from dense_tracker import DenseFlowTracker
from clusterizer import Clusterizer
from object_detector import ObjectDetector
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
    parser.add_argument("--videofile", type=str, default="video/test.mp4",
                        help="Path to the videofile")
    parser.add_argument("--camid", type=int, default=0,
                        help="Id of webcamera")
    parser.add_argument("--roi_tracker", type=str, default="kcf",
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
    clusters_cnt = 4
    clusterizer = Clusterizer(clusters_cnt, debug=True)
    detector = ObjectDetector()

    while True:
        if frame is not None:
            prev_frame = frame.copy()
        ret, frame = cap.read()
        if not ret:
            break
        if cropper:
            frame = cropper(frame)
        visualization_frame = frame.copy()

        if roi_init_bbox:
            prev_roi_bbox = box
            (success, box) = tracker.update(frame)
            if success:
                (roi_x, roi_y, roi_w, roi_h) = [int(v) for v in box]
                cv2.rectangle(visualization_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

                common_roi = find_common_roi(box, prev_roi_bbox)
                if common_roi is not None:
                    if frame is not None and prev_frame is not None:
                        x, y, w, h = common_roi
                        prev_subframe = prev_frame[y:y+h, x:x+w]
                        subframe = frame[y:y+h, x:x+w]
                        mag, ang = dense_tracker(prev_subframe, subframe)
                        labels, clustered_img = clusterizer(mag, ang)
                        objects = detector(labels, clusters_cnt)
                        for obj_x, obj_y, obj_w, obj_h in objects:
                            cv2.rectangle(clustered_img, (obj_x, obj_y), (obj_x+obj_w, obj_y+obj_h), (255, 0, 0), 2)
                            cv2.rectangle(visualization_frame, (roi_x + obj_x, roi_y + obj_y),
                                          (roi_x + obj_x + obj_w, roi_y + obj_y + obj_h), (0, 0, 255), 1)
                        cv2.imshow('Clusterized', clustered_img)
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
