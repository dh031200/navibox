from trackers.ocsort.ocsort import OCSort
from trackers.bytetrack.byte_tracker import BYTETracker


def create_tracker(tracker_type):
    if tracker_type == 'ocsort':
        ocsort = OCSort(
            det_thresh=0.45,
            iou_threshold=0.2,
            use_byte=False 
        )
        return ocsort
    elif tracker_type == 'bytetrack':
        bytetracker = BYTETracker(
            track_thresh=0.6,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=30
        )
        return bytetracker
    else:
        print('No such tracker')
        exit()