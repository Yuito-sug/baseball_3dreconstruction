#!/usr/bin/env python3
"""Robust 2D baseball tracking prototype.

Backend 'auto' uses BaseballCV's pretrained YOLO ball_tracking model when
BaseballCV + Ultralytics + weights are available. Otherwise it falls back to
an OpenCV bright-compact-blob detector with temporal direction/continuity gates.
The fallback is not a trained neural network and should be treated as a baseline.

Example:
  python baseballcv_robust_2d_tracker.py input.mp4 --output tracked.mp4 --backend auto
"""
from __future__ import annotations
import argparse, csv, math, os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

@dataclass
class Candidate:
    x: float
    y: float
    area: float
    w: int
    h: int
    conf: float = 0.0

@dataclass
class TrackPoint:
    frame: int
    x: float
    y: float
    status: str
    confidence: float
    source: str

class OpenCVBallDetector:
    """Bright compact-object detector with resolution-aware, aspect-aware ROI."""
    def __init__(self, threshold: int = 185, saturation_max: int = 135,
                 roi=None):
        self.threshold = threshold
        self.saturation_max = saturation_max
        # Optional ROI is expressed in pixels of a 1920x1080 reference frame.
        # None selects a broad ROI based on the actual video's aspect ratio.
        self.roi = roi

    @staticmethod
    def _merge_nearby(comps, radius=12.0):
        # Union-find merges components when the ball is split by a dark net wire.
        if not comps:
            return []
        n = len(comps); parent = list(range(n))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        def union(a,b):
            ra,rb=find(a),find(b)
            if ra!=rb: parent[rb]=ra
        for i in range(n):
            for j in range(i+1,n):
                if math.hypot(comps[i].x-comps[j].x, comps[i].y-comps[j].y) <= radius:
                    union(i,j)
        groups={}
        for i,c in enumerate(comps): groups.setdefault(find(i),[]).append(c)
        out=[]
        for group in groups.values():
            area=sum(c.area for c in group)
            x=sum(c.x*c.area for c in group)/area
            y=sum(c.y*c.area for c in group)/area
            x1=min(c.x-c.w/2 for c in group); x2=max(c.x+c.w/2 for c in group)
            y1=min(c.y-c.h/2 for c in group); y2=max(c.y+c.h/2 for c in group)
            w=max(1,int(round(x2-x1))); h=max(1,int(round(y2-y1)))
            out.append(Candidate(x,y,area,w,h,max(c.conf for c in group)))
        return out

    def detect(self, frame):
        H, W = frame.shape[:2]
        if self.roi is None:
            # Portrait/behind-home-plate views place the pitcher and ball on the
            # right half; landscape broadcast views need a wider central ROI.
            if W / max(H, 1) < 0.95:
                x1, y1, x2, y2 = int(0.40*W), int(0.30*H), W, int(0.90*H)
            else:
                # Preserve the original reference-view ROI, scaled to this frame size.
                x1, y1 = int(780*W/1920), int(170*H/1080)
                x2, y2 = int(1230*W/1920), int(780*H/1080)
        else:
            rx1, ry1, rx2, ry2 = self.roi
            x1, y1 = int(rx1*W/1920), int(ry1*H/1080)
            x2, y2 = int(rx2*W/1920), int(ry2*H/1080)
        # Clamp all coordinates. Empty crops are reported as no candidates rather
        # than passed to cvtColor (which raises !_src.empty()).
        x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
        y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
        if x2 <= x1 or y2 <= y1:
            return []
        crop=frame[y1:y2,x1:x2]
        if crop.size == 0:
            return []
        hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
        mask=((hsv[:,:,2] > self.threshold) & (hsv[:,:,1] < self.saturation_max)).astype(np.uint8)*255
        # Connected components provide compact bright candidates. Filtering is deliberately
        # permissive; temporal tracking and trajectory gates reject most distractors.
        n, labels, stats, centroids=cv2.connectedComponentsWithStats(mask,8)
        comps=[]
        for i in range(1,n):
            x,y,w,h,area=stats[i]
            cx,cy=centroids[i]
            if not (18 <= area <= 420 and 5 <= w <= 30 and 4 <= h <= 30):
                continue
            if not (0.28 <= (w/max(h,1)) <= 5.0):
                continue
            cx+=x1; cy+=y1
            # Prefer roundish, compact components whose area looks ball-sized.
            aspect=min(w/max(h,1),h/max(w,1))
            size_score=math.exp(-abs(math.log(max(area,1)/125.0)))
            shape_score=math.sqrt(aspect)
            conf=float(np.clip(0.25+0.48*size_score+0.27*shape_score,0,0.99))
            comps.append(Candidate(float(cx),float(cy),float(area),int(w),int(h),conf))
        return self._merge_nearby(comps, radius=12.0)

class BaseballCVYoloDetector:
    """Optional BaseballCV pretrained YOLO backend, following the project README API."""
    def __init__(self, conf=0.005, imgsz=1280):
        from baseballcv.functions import LoadTools
        from ultralytics import YOLO
        path=LoadTools().load_model('ball_tracking')
        self.model=YOLO(path)
        self.conf=conf; self.imgsz=imgsz
    def detect(self, frame):
        result=self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        out=[]
        if result.boxes is None: return out
        for b in result.boxes:
            x1,y1,x2,y2=map(float,b.xyxy[0].cpu().numpy())
            conf=float(b.conf[0].cpu().numpy())
            # Keep small objects for ball tracking; large boxes are likely false positives.
            w=x2-x1; h=y2-y1
            if w<3 or h<3 or w>70 or h>70: continue
            out.append(Candidate((x1+x2)/2,(y1+y2)/2,w*h,int(round(w)),int(round(h)),conf))
        return out

def mean_frame_difference(a,b):
    return float(np.mean(cv2.absdiff(a,b)))

def predict_position(history, next_unique_index, direction_x=1, scale_x=1.0, scale_y=1.0):
    """Predict next center using recent observations and camera-dependent direction."""
    if len(history)==1:
        p=history[-1]
        return np.array([p.x + direction_x*17.0*scale_x, p.y + 32.0*scale_y])
    xy=np.array([[p.x,p.y] for p in history[-5:]],float)
    t=np.arange(len(history)-len(xy),len(history),dtype=float)
    future=float(len(history))
    velocity=xy[-1]-xy[-2]
    linear=xy[-1]+velocity
    if len(xy)>=3:
        deg=min(2,len(xy)-1)
        pred=[]
        for axis in range(2):
            co=np.polyfit(t,xy[:,axis],deg=deg)
            pred.append(float(np.polyval(co,future)))
        quad=np.array(pred)
        return 0.65*linear+0.35*quad
    return linear

def track_frames(frames, backend='auto', force_opencv=False):
    if not frames:
        raise RuntimeError('Input video contains no readable frames.')
    H, W = frames[0].shape[:2]
    portrait = (W / max(H, 1)) < 0.95
    direction_x = -1 if portrait else 1
    scale_x = W / 1920.0
    scale_y = H / 1080.0

    model=None; backend_used='opencv-classical'
    if backend in ('auto','baseballcv-yolo') and not force_opencv:
        try:
            model=BaseballCVYoloDetector()
            backend_used='BaseballCV YOLO ball_tracking'
        except Exception as e:
            if backend == 'baseballcv-yolo':
                raise RuntimeError(f'BaseballCV YOLO backend unavailable: {e}') from e
            model=None
    if model is None:
        model=OpenCVBallDetector(threshold=185)
    candidates=[model.detect(f) for f in frames]

    # Camera-dependent release search zone. These are normalized coordinates,
    # not hard-coded 1920x1080 pixel positions.
    if portrait:
        release_zone=(0.58*W, 0.47*H, 0.78*W, 0.67*H)
    else:
        # Preserve the original landscape release zone, now expressed as ratios.
        release_zone=(820*W/1920, 190*H/1080, 885*W/1920, 250*H/1080)
    rx1, ry1, rx2, ry2 = release_zone
    seed=None
    for fi,cs in enumerate(candidates):
        matches=[]
        for c in cs:
            if rx1 <= c.x <= rx2 and ry1 <= c.y <= ry2 and 25 <= c.area <= 300:
                # Favor compact, near-white candidates near the expected release region.
                cx0=(rx1+rx2)/2; cy0=(ry1+ry2)/2
                scale=max(0.01, min(W,H)/1080.0)
                score=(abs(c.x-cx0)/max((rx2-rx1)/2,1) +
                       abs(c.y-cy0)/max((ry2-ry1)/2,1) +
                       0.15*abs(math.log(max(c.area,1)/100.0)))
                matches.append((score,c))
        if matches:
            _,c=min(matches,key=lambda z:z[0]); seed=(fi,c); break
    if seed is None:
        raise RuntimeError(
            'Could not find a release-zone candidate. The clip may use a different camera angle; '
            'adjust the normalized release zone in track_frames() or use --backend baseballcv-yolo.'
        )
    start_frame, sc=seed
    points=[]; history=[]; unique_idx=0
    first=TrackPoint(start_frame,sc.x,sc.y,'detected-release',sc.conf,backend_used)
    points.append(first); history.append(first)
    last_idx=start_frame; last_point=first
    for fi in range(start_frame+1,len(frames)):
        delta=mean_frame_difference(frames[fi-1],frames[fi])
        if delta < 0.45:
            pt=TrackPoint(fi,last_point.x,last_point.y,'duplicate-frame-copy',last_point.confidence,backend_used)
            points.append(pt); last_idx=fi; last_point=pt
            continue
        cs=candidates[fi]
        pred=predict_position(history,unique_idx+1,direction_x,scale_x,scale_y)
        viable=[]
        prev=history[-1]
        for c in cs:
            dx=c.x-prev.x; dy=c.y-prev.y
            # Infer the horizontal direction from image aspect ratio; vertical
            # motion is downward in the two supported broadcast views.
            if direction_x*dx < -2.0*scale_x or dy < -3.0*scale_y:
                continue
            if portrait:
                # Behind-home-plate portrait clips in this sample have a much
                # slower screen-space displacement than the original landscape clip.
                # This rejects catcher/helmet highlights that jump too far per frame.
                if abs(dx) > 80*scale_x or dy > 65*scale_y:
                    continue
            else:
                if abs(dx) > 150*max(scale_x,scale_y) or dy > 180*max(scale_x,scale_y):
                    continue
            d=float(np.linalg.norm(np.array([c.x,c.y])-pred))
            gate=max(32.0,155.0*max(scale_x,scale_y)) if len(history)<5 else max(24.0,115.0*max(scale_x,scale_y))
            if d > gate: continue
            # Require the candidate to move in the same general direction.
            if abs(dx) > 1 and abs(dy) > 1:
                ratio=abs(dx)/max(abs(dy),1)
                if not (0.05 <= ratio <= 4.0): continue
            score=d - 8.0*c.conf + 0.10*abs(math.log(max(c.area,1)/100.0))
            viable.append((score,c,d))
        if not viable:
            # A missed frame is tolerated; do not jump to another bright object.
            continue
        score,c,d=min(viable,key=lambda z:z[0])
        pt=TrackPoint(fi,c.x,c.y,'detected',float(c.conf),backend_used)
        points.append(pt); history.append(pt); last_idx=fi; last_point=pt; unique_idx+=1

    distinct=[p for p in points if p.status in ('detected','detected-release')]
    if not distinct:
        raise RuntimeError('No ball candidates were tracked after release detection.')
    last_obs=distinct[-1]
    # Estimate continuation from recent visible motion. Keep the direction sign
    # and use resolution-scaled bounds; endpoint is NOT asserted to be a catch.
    if len(distinct)>=2:
        v=np.array([distinct[-1].x-distinct[-2].x,distinct[-1].y-distinct[-2].y],float)
    else:
        v=np.array([direction_x*12.0*scale_x,18.0*scale_y])
    min_vx=1.0*scale_x; max_vx=45.0*scale_x
    v[0]=direction_x*float(np.clip(abs(v[0]),min_vx,max_vx))
    v[1]=float(np.clip(v[1],2.0*scale_y,75.0*scale_y))
    # Occlusion begins after the last actual observation. Stop at frame bounds.
    occ=[]
    base=np.array([last_obs.x,last_obs.y],float)
    start_occ=last_obs.frame+1
    prior=base.copy()
    for fi in range(start_occ,min(len(frames),start_occ+3)):
        if mean_frame_difference(frames[fi-1],frames[fi]) < 0.45:
            pos=prior.copy(); status='occlusion-estimate-duplicate'
        else:
            pos=prior+v; status='occlusion-estimate'
        pos[0]=float(np.clip(pos[0],0,W-1)); pos[1]=float(np.clip(pos[1],0,H-1))
        occ.append((fi,pos,status)); prior=pos
    return points, occ, start_frame, last_obs.frame, backend_used, candidates

def status_prev_dup(frames, fi):
    return mean_frame_difference(frames[fi-1],frames[fi]) < 0.45

def draw_dashed_line(img,p1,p2,color,thickness=3,dash=12,gap=8):
    p1=np.array(p1,float); p2=np.array(p2,float); vec=p2-p1; length=float(np.linalg.norm(vec))
    if length<1:return
    unit=vec/length; d=0.0
    while d<length:
        e=min(d+dash,length)
        a=tuple(np.round(p1+unit*d).astype(int)); b=tuple(np.round(p1+unit*e).astype(int))
        cv2.line(img,a,b,color,thickness,cv2.LINE_AA); d=e+gap

def render(input_path, output_path, csv_path, backend='auto'):
    cap=cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open input video: {input_path}')
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0; W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames=[]
    while True:
        ok,f=cap.read()
        if not ok:break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f'No frames could be decoded from: {input_path}')
    points,occ,start,last,backend_used,candidates=track_frames(frames,backend=backend)
    byframe={p.frame:p for p in points}
    occmap={i:(xy,st) for i,xy,st in occ}
    # Observed/detected points as a piecewise path; draw a solid green segment only between actual observed detections.
    actual=[p for p in points if p.status in ('detected','detected-release')]
    unique_actual=[]
    for p in actual:
        if unique_actual and np.linalg.norm(np.array([p.x,p.y])-np.array([unique_actual[-1].x,unique_actual[-1].y]))<0.1:
            continue
        unique_actual.append(p)
    # dashed inference joins last visible location to estimated points
    fourcc=cv2.VideoWriter_fourcc(*'mp4v'); writer=cv2.VideoWriter(output_path,fourcc,fps,(W,H))
    if not writer.isOpened():
        raise RuntimeError(f'Could not create output video: {output_path}')
    font=cv2.FONT_HERSHEY_SIMPLEX
    for i,src in enumerate(frames):
        vis=src.copy()
        hist=[p for p in unique_actual if p.frame<=i]
        if len(hist)>=2:
            pts=np.array([[int(p.x),int(p.y)] for p in hist],np.int32)
            cv2.polylines(vis,[pts],False,(0,220,0),3,cv2.LINE_AA)
        # Keep trajectory point on duplicated frame, but not falsely label it as new observation.
        if i in byframe:
            p=byframe[i]
            if p.status=='detected-release':
                cv2.circle(vis,(int(p.x),int(p.y)),20,(255,255,0),3,cv2.LINE_AA)
                cv2.putText(vis,'RELEASE',(int(p.x)+22,int(p.y)-20),font,0.7,(255,255,0),2,cv2.LINE_AA)
            elif p.status in ('detected','duplicate-frame-copy'):
                cv2.rectangle(vis,(int(p.x)-11,int(p.y)-11),(int(p.x)+11,int(p.y)+11),(0,0,255),2,cv2.LINE_AA)
                cv2.circle(vis,(int(p.x),int(p.y)),3,(0,0,255),-1,cv2.LINE_AA)
                if p.status=='duplicate-frame-copy':
                    cv2.putText(vis,'duplicate-frame copy',(int(p.x)+14,int(p.y)+22),font,0.48,(0,0,255),1,cv2.LINE_AA)
        if i in occmap:
            (x,y),st=occmap[i]; xy=(int(x),int(y))
            if i>start:
                prev_occ=[(j,q) for j,q,s in occ if j<i]
                if prev_occ:
                    px,py=prev_occ[-1][1]; draw_dashed_line(vis,(int(px),int(py)),xy,(0,165,255),3)
                else:
                    draw_dashed_line(vis,(int(last if False else unique_actual[-1].x),int(unique_actual[-1].y)),xy,(0,165,255),3)
            cv2.circle(vis,xy,11,(0,165,255),3,cv2.LINE_AA)
            cv2.putText(vis,'inferred',(xy[0]+14,xy[1]-10),font,0.58,(0,165,255),2,cv2.LINE_AA)
            if st.startswith('occlusion-estimate'):
                cv2.drawMarker(vis,xy,(0,165,255),cv2.MARKER_CROSS,26,2,cv2.LINE_AA)
        # Legend
        cv2.rectangle(vis,(20,18),(960,122),(0,0,0),-1)
        cv2.putText(vis,'2D ball tracker | '+backend_used,(36,47),font,0.67,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(vis,'Red boxes: ball candidate selected by detector',(36,74),font,0.59,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(vis,'Green: observed track | Orange dashed: estimated continuation',(36,99),font,0.59,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(vis,f'Frame {i}/{len(frames)-1}',(W-210,38),font,0.58,(255,255,255),1,cv2.LINE_AA)
        writer.write(vis)
    writer.release()
    # CSV record every frame: observed detection, duplicate-frame copy, occlusion estimate, or no ball track.
    allpos={p.frame:p for p in points}
    for fi,(xy,st) in occmap.items():
        allpos[fi]=TrackPoint(fi,float(xy[0]),float(xy[1]),st,0.25,'trajectory extrapolation')
    with open(csv_path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['frame','time_sec','x_px','y_px','status','confidence','source'])
        for i in range(len(frames)):
            p=allpos.get(i)
            if p is None: w.writerow([i,round(i/fps,4),'','','not_tracked','',''])
            else: w.writerow([i,round(i/fps,4),round(p.x,2),round(p.y,2),p.status,round(p.confidence,3),p.source])
    return points,occ,backend_used,fps,W,H

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('input',help='input mp4 path')
    ap.add_argument('--output',default='baseball_2d_track.mp4')
    ap.add_argument('--csv',default='baseball_2d_track.csv')
    ap.add_argument('--backend',choices=['auto','opencv','baseballcv-yolo'],default='auto')
    args=ap.parse_args()
    points,occ,backend,fps,W,H=render(args.input,args.output,args.csv,args.backend)
    print(f'backend={backend} size={W}x{H} fps={fps} tracked={len(points)} occlusion_estimates={len(occ)}')
    print('detected:',[(p.frame,round(p.x,1),round(p.y,1),p.status) for p in points if p.status in ('detected','detected-release')])
    print('occlusion:',[(i,xy.tolist(),st) for i,xy,st in occ])
    print('NOTE: occlusion coordinates are estimates, not detections. OpenCV fallback is not a learned model.')
if __name__=='__main__': main()
