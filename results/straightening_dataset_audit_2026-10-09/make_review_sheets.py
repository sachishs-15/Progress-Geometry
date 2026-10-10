"""Reproducible, read-only video sampling and contact sheets for this audit."""
import csv
import json
from pathlib import Path
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
RNG = random.Random(20261009)
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)


def rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def select():
    k = rows(PROJECT / 'data/kinetics-400/k400_train_full.csv')
    by_class = {}
    for r in k:
        by_class.setdefault(r['label'], []).append(r)
    # One per label, then a random sample of 100 labels: a class-balanced audit.
    labels = sorted(by_class)
    picked = RNG.sample(labels, 100)
    ks = [next((r for r in RNG.sample(by_class[label],len(by_class[label])) if Path(r['video_path']).is_file()),None) for label in picked]
    ks = [r for r in ks if r is not None]
    e = rows(PROJECT / 'data/ego4dgoal/ego4dgoal_val.csv')
    by_goal = {}
    for r in e:
        by_goal.setdefault(r['goal_category'], []).append(r)
    # Cover distinct goal families, then fill to 80 clips.
    es = [next((r for r in RNG.sample(v,len(v)) if Path(r['video_path']).is_file()),None) for v in by_goal.values()]
    es = [r for r in es if r is not None]
    seen = {x['clip_id'] for x in es}
    pool = [x for x in e if x['clip_id'] not in seen]
    es += [r for r in RNG.sample(pool,len(pool)) if Path(r['video_path']).is_file()][:80-len(es)]
    RNG.shuffle(es)
    by_video = {}
    for r in e:
        by_video.setdefault(r['video_uid'], []).append(r)
    # Long-recording sample, spread across goal families.
    whole = []
    used = set()
    for goal in sorted(by_goal):
        options = sorted({r['video_uid'] for r in by_goal[goal] if r['video_uid'] in by_video})
        if options:
            uid = RNG.choice(options)
            if uid not in used:
                whole.append((uid, by_video[uid]))
                used.add(uid)
    if len(whole) > 24:
        whole = RNG.sample(whole, 24)
    for uid in RNG.sample(sorted(set(by_video)-used), min(24-len(whole), len(set(by_video)-used))):
        whole.append((uid, by_video[uid]))
    return ks, es, whole


def frames(path, n=6):
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    duration = count/fps if fps > 0 else 0
    out=[]
    for ix in np.linspace(max(0, int(count*0.03)), max(0,int(count*0.97)-1), n, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(ix))
        ok, frame = cap.read()
        if ok:
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            out.append(Image.fromarray(frame))
        else:
            out.append(None)
    cap.release()
    return out,duration


def sheets(kind, items):
    d=ROOT/kind
    d.mkdir(exist_ok=True)
    metadata=[]
    for si in range(0,len(items),10):
        sheet=Image.new('RGB',(1512,10*156),'white')
        draw=ImageDraw.Draw(sheet)
        for j,item in enumerate(items[si:si+10]):
            idx=si+j
            path=Path(item['path'])
            ff,dur=frames(path)
            y=j*156
            title=f"{kind.upper()} {idx:03d} | {item['label'][:85]} | {dur:.1f}s"
            draw.text((7,y+2),title,fill='black',font=FONT)
            for t,frame in enumerate(ff):
                x=t*252+2
                if frame is not None:
                    frame.thumbnail((248,124))
                    sheet.paste(frame,(x+(248-frame.width)//2,y+22))
                draw.text((x+3,y+143),f"{(t+.03)*(dur*.94)/5:.1f}s",fill='black',font=FONT)
            metadata.append({'kind':kind,'index':idx,'path':str(path),'label':item['label'],'duration_s':dur,**item.get('extra',{})})
        sheet.save(d/f'sheet_{si//10:02d}.jpg',quality=88)
    with open(ROOT/f'{kind}_sample.json','w') as f: json.dump(metadata,f,indent=2)


if __name__=='__main__':
    ks,es,whole=select()
    sheets('kinetics',[{'path':r['video_path'],'label':r['label']} for r in ks])
    sheets('ego_leaf',[{'path':r['video_path'],'label':r['leaf_step_description'] or r['leaf_step_category'],'extra':{'goal':r['goal_category'],'relevance':r['leaf_is_relevant'],'video_uid':r['video_uid']}} for r in es])
    sheets('ego_whole',[{'path':f"/mnt/rdata7/proprogress/data/ego4dgoal/videos/{uid}.mp4",'label':rs[0]['goal_category'],'extra':{'video_uid':uid,'n_annotated_clips':len(rs)}} for uid,rs in whole])
