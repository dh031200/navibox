from core.detector import Detector
from core.distortManager import DistortManager
from core.videoManager import VideoManager

video_manager = VideoManager('src/sample.mp4')
distort_manager = DistortManager(video_manager.width, video_manager.height, 'src/cam_calib_v2.pkl')
detector = Detector('src/yolov7-navibox-best-221121.pt')

while True:
    ret, frame = video_manager.read()
    if not ret:
        break
    img = distort_manager.un_distort(frame)
    results = detector.detect(img)
    video_manager.draw(img, results, detector.names)
    video_manager.write(img)
video_manager.release()
