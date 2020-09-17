import cv2
import numpy as np


class Cropper:

    def __init__(self, roi):
        assert len(roi) == 4
        self.x, self.y, self.w, self.h = roi
        assert type(self.w) == int
        assert type(self.h) == int
        return

    def __call__(self, *args, **kwargs):
        img = args[0]
        assert type(img) == np.ndarray
        assert 2 <= len(img.shape) <= 3
        xsz, ysz = img.shape[1], img.shape[0]
        x, y = self.x, self.y
        if (x is None) or (y is None):
            x = int((xsz - self.w) / 2)
            y = int((ysz - self.h) / 2)
        x1 = min(xsz - 1, max(0, x))
        x2 = min(xsz - 1, max(0, x + self.w - 1))
        y1 = min(ysz - 1, max(0, y))
        y2 = min(ysz - 1, max(0, y + self.h - 1))
        return img[y1:y2+1, x1:x2+1]


class DenseFlowTracker:

    def __init__(self):
        self.optflow_config = dict(prev=None, next=None, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                   poly_n=5, poly_sigma=1.2, flags=0)
        pass

    def __str__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def process(self, prev_frame, next_frame, mask=None):
        assert type(prev_frame) == np.ndarray
        assert type(next_frame) == np.ndarray
        assert len(prev_frame.shape) == 2
        assert len(next_frame.shape) == 2
        self.optflow_config["prev"] = prev_frame.copy()
        self.optflow_config["next"] = next_frame.copy()
        flow = cv2.calcOpticalFlowFarneback(**self.optflow_config)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return mag, ang


if __name__ == "__main__":
    cap = cv2.VideoCapture("test.mp4")

    tracker = DenseFlowTracker()
    cropper = Cropper(roi=(None, None, 600, 600))

    ret, frame1 = cap.read()
    frame1 = cropper(frame1)
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                          (frame1.shape[1], frame1.shape[0]))

    ret = True
    while ret:
        ret, frame2 = cap.read()
        frame2 = cropper(frame2)
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