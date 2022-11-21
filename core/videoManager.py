import cv2


class VideoManager:
    def __init__(self, src, output='out.mp4'):
        self.cap = cv2.VideoCapture(src)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))
        self.colors = ((255, 0, 0), (0, 0, 255), (255, 50, 255))

    def draw(self, img, results, names):
        for t in results:
            cls = int(t.cls)
            label = f'{t.track_id}_{names[cls]}_{t.score:.2f}'
            self.plot_one_box(t.tlbr, int(cls), img, label=label, line_thickness=2)

    def plot_one_box(self, x, cls, img, label=None, line_thickness=3):
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, self.colors[cls], thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, self.colors[cls], -1, cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def read(self):
        return self.cap.read()

    def write(self, img):
        self.writer.write(img)

    def release(self):
        self.cap.release()
        self.writer.release()
