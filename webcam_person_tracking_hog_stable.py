import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple

@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

@dataclass
class Detection:
    bbox: BBox
    score: float

def iou(a: BBox, b: BBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.width() * a.height() + b.width() * b.height() - inter
    return float(inter / union) if union > 0 else 0.0

def iou_matrix_tracks_dets(tracks: List["Track"], dets: List[Detection]) -> np.ndarray:
    if len(tracks) == 0 or len(dets) == 0:
        return np.zeros((len(tracks), len(dets)), dtype=np.float32)
    M = np.zeros((len(tracks), len(dets)), dtype=np.float32)
    for i, t in enumerate(tracks):
        tb = t.to_bbox()
        for j, d in enumerate(dets):
            M[i, j] = iou(tb, d.bbox)
    return M

def nms_detections(dets: List[Detection], iou_thr: float = 0.5) -> List[Detection]:
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    keep: List[Detection] = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if iou(best.bbox, d.bbox) < iou_thr]
    return keep

def hungarian(cost: np.ndarray):
    cost = cost.copy().astype(np.float64)
    n, m = cost.shape
    N = max(n, m)
    pad_val = cost.max() + 1.0 if cost.size else 1.0
    C = np.full((N, N), pad_val, dtype=np.float64)
    C[:n, :m] = cost

    u = np.zeros(N + 1)
    v = np.zeros(N + 1)
    p = np.zeros(N + 1, dtype=int)
    way = np.zeros(N + 1, dtype=int)

    for i in range(1, N + 1):
        p[0] = i
        j0 = 0
        minv = np.full(N + 1, np.inf)
        used = np.zeros(N + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, N + 1):
                if not used[j]:
                    cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, N + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assign = []
    for j in range(1, N + 1):
        i = p[j]
        r = i - 1
        c = j - 1
        if r < n and c < m:
            assign.append((r, c))
    return assign

def assign_with_hungarian(cost: np.ndarray, cost_limit: float):
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    pairs = hungarian(cost)
    matches = []
    used_r = set()
    used_c = set()
    for r, c in pairs:
        if cost[r, c] <= cost_limit:
            matches.append((r, c))
            used_r.add(r)
            used_c.add(c)
    unmatched_r = [r for r in range(n) if r not in used_r]
    unmatched_c = [c for c in range(m) if c not in used_c]
    return matches, unmatched_r, unmatched_c

class KalmanFilterCV:
    def __init__(self):
        ndim, dt = 4, 1.0
        self.F = np.eye(2 * ndim, dtype=np.float64)
        for i in range(ndim):
            self.F[i, ndim + i] = dt
        self.H = np.eye(ndim, 2 * ndim, dtype=np.float64)
        self.std_pos = 1.0
        self.std_vel = 0.5
        self.std_meas = 1.0
    @staticmethod
    def bbox_to_z(b: BBox) -> np.ndarray:
        w = max(1e-6, b.x2 - b.x1)
        h = max(1e-6, b.y2 - b.y1)
        cx = b.x1 + w / 2.0
        cy = b.y1 + h / 2.0
        a = w / h
        return np.array([cx, cy, a, h], dtype=np.float64)
    @staticmethod
    def z_to_bbox(z: np.ndarray) -> BBox:
        cx, cy, a, h = float(z[0]), float(z[1]), float(z[2]), float(z[3])
        h = max(1e-6, h)
        a = max(1e-6, a)
        w = a * h
        return BBox(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
    def initiate(self, bbox: BBox):
        z = self.bbox_to_z(bbox)
        x = np.zeros((8,), dtype=np.float64)
        x[:4] = z
        P = np.eye(8, dtype=np.float64)
        P[:4, :4] *= (10.0 * self.std_pos) ** 2
        P[4:, 4:] *= (10.0 * self.std_vel) ** 2
        return x, P
    def predict(self, x, P):
        Q = np.eye(8, dtype=np.float64)
        Q[:4, :4] *= self.std_pos ** 2
        Q[4:, 4:] *= self.std_vel ** 2
        x = self.F @ x
        P = self.F @ P @ self.F.T + Q
        return x, P
    def update(self, x, P, bbox: BBox):
        z = self.bbox_to_z(bbox)
        R = np.eye(4, dtype=np.float64) * (self.std_meas ** 2)
        y = z - (self.H @ x)
        S = self.H @ P @ self.H.T + R
        K = P @ self.H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(8) - K @ self.H) @ P
        return x, P

class TrackState(Enum):
    Tracked = auto()
    Lost = auto()
    Removed = auto()

class Track:
    _next_id = 1
    def __init__(self, det: Detection, frame_id: int, kf: KalmanFilterCV):
        self.track_id = Track._next_id
        Track._next_id += 1
        self.kf = kf
        self.x, self.P = self.kf.initiate(det.bbox)
        self.score = det.score
        self.state = TrackState.Tracked
        self.start_frame = frame_id
        self.last_frame = frame_id
        self.time_since_update = 0
        self.hits = 1
    def predict(self):
        self.x, self.P = self.kf.predict(self.x, self.P)
        self.time_since_update += 1
    def update(self, det: Detection, frame_id: int):
        self.x, self.P = self.kf.update(self.x, self.P, det.bbox)
        self.score = det.score
        self.state = TrackState.Tracked
        self.last_frame = frame_id
        self.time_since_update = 0
        self.hits += 1
    def mark_lost(self):
        self.state = TrackState.Lost
    def mark_removed(self):
        self.state = TrackState.Removed
    def to_bbox(self):
        return self.kf.z_to_bbox(self.x[:4])

class BYTETracker:
    def __init__(self, track_thresh=0.25, low_thresh=0.05, match_iou_thresh=0.15, max_time_lost=18, det_nms_iou=0.5, min_hits=2):
        self.track_thresh = float(track_thresh)
        self.low_thresh = float(low_thresh)
        self.match_iou_thresh = float(match_iou_thresh)
        self.max_time_lost = int(max_time_lost)
        self.det_nms_iou = float(det_nms_iou)
        self.min_hits = int(min_hits)
        self.frame_id = 0
        self.kf = KalmanFilterCV()
        self.tracked = []
        self.lost = []
        self.removed = []
    def _split_dets(self, dets):
        dets = [d for d in dets if d.score >= self.low_thresh]
        dets = nms_detections(dets, iou_thr=self.det_nms_iou)
        high = [d for d in dets if d.score >= self.track_thresh]
        low = [d for d in dets if self.low_thresh <= d.score < self.track_thresh]
        return high, low
    def update(self, detections):
        self.frame_id += 1
        det_high, det_low = self._split_dets(detections)
        for t in self.tracked:
            t.predict()
        for t in self.lost:
            t.predict()
        candidates = [t for t in self.tracked if t.state == TrackState.Tracked] + [t for t in self.lost if t.state == TrackState.Lost]
        matched_ids = set()
        ious1 = iou_matrix_tracks_dets(candidates, det_high)
        cost1 = 1.0 - ious1
        matches1, un_cand, un_high = assign_with_hungarian(cost1, cost_limit=1.0 - self.match_iou_thresh)
        for ci, di in matches1:
            candidates[ci].update(det_high[di], self.frame_id)
            matched_ids.add(candidates[ci].track_id)
        remain_tracks = [candidates[i] for i in un_cand]
        ious2 = iou_matrix_tracks_dets(remain_tracks, det_low)
        cost2 = 1.0 - ious2
        matches2, _, _ = assign_with_hungarian(cost2, cost_limit=1.0 - self.match_iou_thresh)
        for ri, di in matches2:
            remain_tracks[ri].update(det_low[di], self.frame_id)
            matched_ids.add(remain_tracks[ri].track_id)
        still_tracked = []
        new_lost = []
        for t in self.tracked:
            if t.track_id in matched_ids:
                still_tracked.append(t)
            else:
                t.mark_lost()
                new_lost.append(t)
        kept_lost = []
        for t in self.lost + new_lost:
            if t.track_id in matched_ids:
                still_tracked.append(t)
                continue
            if self.frame_id - t.last_frame > self.max_time_lost:
                t.mark_removed()
                self.removed.append(t)
            else:
                kept_lost.append(t)
        for di in un_high:
            still_tracked.append(Track(det_high[di], self.frame_id, self.kf))
        self.tracked = [t for t in still_tracked if t.state == TrackState.Tracked]
        self.lost = [t for t in kept_lost if t.state == TrackState.Lost]
        return [t for t in self.tracked if t.hits >= self.min_hits]

def make_hog_detector():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog

def detect_people_hog(frame: np.ndarray, hog) -> List[Detection]:
    target_w = 960
    h0, w0 = frame.shape[:2]
    scale_img = target_w / max(1, w0)
    resized = cv2.resize(frame, (int(w0 * scale_img), int(h0 * scale_img)))
    rects, weights = hog.detectMultiScale(
        resized,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.03,
        useMeanshiftGrouping=False,
    )
    dets: List[Detection] = []
    for (x, y, w, h), weight in zip(rects, weights):
        x1 = x - 0.05 * w
        y1 = y - 0.08 * h
        x2 = x + 1.05 * w
        y2 = y + 1.08 * h
        x1 /= scale_img
        y1 /= scale_img
        x2 /= scale_img
        y2 /= scale_img
        bbox = BBox(
            max(0.0, x1),
            max(0.0, y1),
            min(float(w0 - 1), x2),
            min(float(h0 - 1), y2),
        )
        score = float(max(0.05, min(0.99, weight / 2.0)))
        if bbox.width() >= 24 and bbox.height() >= 48:
            dets.append(Detection(bbox, score))
    return nms_detections(dets, iou_thr=0.45)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Không mở được camera.")
    hog = make_hog_detector()
    tracker = BYTETracker(track_thresh=0.25, low_thresh=0.05, match_iou_thresh=0.15, max_time_lost=18, det_nms_iou=0.5, min_hits=2)
    frame_id = 0
    fps_show = 0.0
    t_prev = cv2.getTickCount()
    print("Dang chay webcam voi HOG person detector + tracking...")
    print("q=thoat | s=luu anh | r=reset tracker")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        dets = detect_people_hog(frame, hog)
        tracks = tracker.update(dets)
        vis = frame.copy()
        for d in dets:
            x1, y1, x2, y2 = map(int, [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
        for t in tracks:
            bb = t.to_bbox()
            x1, y1, x2, y2 = map(int, [bb.x1, bb.y1, bb.x2, bb.y2])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"ID {t.track_id}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        t_now = cv2.getTickCount()
        dt = (t_now - t_prev) / cv2.getTickFrequency()
        t_prev = t_now
        if dt > 0:
            cur_fps = 1.0 / dt
            fps_show = cur_fps if fps_show == 0 else 0.9 * fps_show + 0.1 * cur_fps
        cv2.putText(vis, f"FPS: {fps_show:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, f"Detections: {len(dets)} | Tracks: {len(tracks)} | Lost: {len(tracker.lost)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
        cv2.putText(vis, "Green = stable track ID | Yellow = person detection", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
        cv2.putText(vis, "q=quit  s=save  r=reset tracker", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
        cv2.imshow("Webcam Person Tracking (HOG + ByteTrack-style)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            out_name = f"hog_track_{frame_id:06d}.jpg"
            cv2.imwrite(out_name, vis)
            print("Da luu:", out_name)
        elif key == ord("r"):
            tracker = BYTETracker(track_thresh=0.25, low_thresh=0.05, match_iou_thresh=0.15, max_time_lost=18, det_nms_iou=0.5, min_hits=2)
            print("Da reset tracker.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # pip install numpy opencv-python
    main()
