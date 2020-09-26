import cv2
import numpy as np


class DenseFlowTracker:

    def __init__(self):
        self.optflow_config = dict(prev=None, next=None, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                   poly_n=5, poly_sigma=1.2, flags=0)
        pass

    def __str__(self):
        return "DenseFlowTracker"

    def __call__(self, *args, **kwargs):
        return self.process(args[0], args[1])

    def process(self, prev_frame, next_frame):
        assert type(prev_frame) == np.ndarray
        assert type(next_frame) == np.ndarray
        p_frame, n_frame = prev_frame.copy(), next_frame.copy()
        if len(p_frame.shape) != 2:
            p_frame = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)
        if len(n_frame.shape) != 2:
            n_frame = cv2.cvtColor(n_frame, cv2.COLOR_BGR2GRAY)
        self.optflow_config["prev"] = p_frame.copy()
        self.optflow_config["next"] = n_frame.copy()
        flow = cv2.calcOpticalFlowFarneback(**self.optflow_config)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return mag, ang


if __name__ == "__main__":
    cap = cv2.VideoCapture("test.mp4")

    tracker = DenseFlowTracker()

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                          (frame1.shape[1], frame1.shape[0]))

    ret = True
    while ret:
        ret, frame2 = cap.read()
        if not ret:
            break

        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        mag, ang = tracker.process(prvs, next)

        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2', rgb)
        cv2.imshow('src', frame2)
        out.write(rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',rgb)
        prvs = next

    cap.release()
    cv2.destroyAllWindows()
    out.release()