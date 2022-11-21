import time
import cv2
import pickle


class DistortManager:
    def __init__(self, width, height, path):
        roi, self.mapx, self.mapy = get_cameramat_dist(path, width, height)
        self.x, self.y, self.width, self.height = roi

    def un_distort(self, img):
        height, width = img.shape[:2]
        return cv2.resize(cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)[self.y:self.y + self.height,
                          self.x:self.x + self.width], (width, height))


def get_cameramat_dist(filename, width, height):
    with open(filename, 'rb') as f:
        mat, dist, rvecs, tvecs = pickle.load(f)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mat, dist, (width, height), 0, (width, height))
        mapx, mapy = cv2.initUndistortRectifyMap(mat, dist, None, newcameramtx, (width, height), 5)
    return roi, mapx, mapy


def main():
    frame_count = 0
    cap = cv2.VideoCapture('sample.mp4')
    height, width = map(int, [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)])
    distortmanager = DistortManager(width, height, '../src/cam_calib_v2.pkl')
    fps = cap.get(cv2.CAP_PROP_FPS)
    # out = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # undistort
        cv2.imshow('frame', distortmanager.un_distort(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # out.write(undistorted_frame)
    cap.release()
    cv2.destroyAllWindows()
    elapsed_time = time.time() - start_time
    print(frame_count / elapsed_time)


if __name__ == "__main__":
    main()
