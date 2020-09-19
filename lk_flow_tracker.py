import numpy as np
import cv2


class OptFlowTracker:

    def __init__(self):
        # params for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)

        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        pass

    def __str__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def process(self, prev_frame, cur_frame, points_to_track=None):
        if points_to_track is None:
            points_to_track = cv2.goodFeaturesToTrack(prev_frame, mask=None, **self.feature_params)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, cur_frame, points_to_track, None, **self.lk_params)
        # Select good points
        good_new = p1[st == 1]
        good_old = points_to_track[st == 1]
        return good_old, good_new


if __name__ == "__main__":
    cap = cv2.VideoCapture('test.mp4')
    tracker = OptFlowTracker()

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    color = np.random.randint(0, 255, (100, 3))
    p2track = None

    while True:
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        good_old, good_new = tracker.process(old_gray, frame_gray, points_to_track=p2track)

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)

        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()

    cv2.destroyAllWindows()
    cap.release()
