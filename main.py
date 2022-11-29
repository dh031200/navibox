from time import time
from multiprocessing import Queue, freeze_support

import cv2

from core.detector import Detector
from core.distortManager import DistortManager
from core.visualizer import VideoManager, Visualizer
from core.povManager import POVManager
from core.config import Config


def main():
    cfg = Config()
    detector = Detector(cfg.model_path)
    video_manager = VideoManager(cfg.source)
    distort_manager = DistortManager(video_manager.width, video_manager.height, cfg.distortion_data)
    pov_manager = POVManager(cfg.camera_space, cfg.world_space)
    mp_queue = Queue()
    visualizer = Visualizer(mp_queue, detector.names, video_manager.width, video_manager.height, video_manager.fps,
                            visualize=True)
    s_map = cv2.imread(cfg.map)
    visualizer.start()
    s_time = time()
    it = time()
    for frame in video_manager:
        # print(f'read : {time() - it}s')
        it = time()
        img = distort_manager.un_distort(frame)
        # print(f'undistortion : {time() - it}s')
        it = time()
        results = detector.detect(img)
        mp_queue.put([img, results])
        # # print(f'detect : {time() - it}s')
        # it = time()
        # canvas = s_map

        # ###########################################################
        # for res in results:
        #     cv2.circle(canvas, pov_manager.coord_transform(res), 3, (0, 0, 255), -1)
        # ###########################################################
        # video_manager.draw(img, results, detector.names)
        # print(f'draw : {time() - it}s')
        # it = time()
        # video_manager.write(img)
        # # print(f'write : {time() - it}s')
        # it = time()

        it = time()
        # print(f'total : {time() - s_time}')
        s_time = time()
        # print('---------------------------')
    mp_queue.put(-1)
    visualizer.join()
    mp_queue.close()
    video_manager.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    freeze_support()
    main()
