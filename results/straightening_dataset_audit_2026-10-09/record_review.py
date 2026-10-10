"""Manual six-frame contact-sheet labels. Run after reviewing the saved sheets."""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent

# Kinetics: S = visible cumulative target state; P = plausible but state not visible;
# T = travel or one-shot phase without a durable task state; R = repetition/reversal;
# U = cut, weak coverage, or label mismatch that prevents judgment.
KINETICS={
 'S':[3,16,35,40,44,62,66,73,74,76,90,96],
 'P':[6,17,21,24,25,32,33,38,39,42,50,56,58,71,72,75,91,98],
 'T':[0,1,5,7,9,10,12,14,18,22,28,29,31,34,41,45,46,48,49,53,59,60,61,63,69,70,83,86,87,92,99],
 'R':[2,4,13,15,23,26,27,30,36,37,43,47,52,57,64,65,67,68,77,78,79,81,82,88,93,95,97],
 'U':[8,11,19,20,51,54,55,80,84,85,89,94],
}

# Ego4D GoalStep: uniform random sample of validation leaf clips (seed 99239).
# Here T denotes a brief one-way event or transfer whose phase is not a durable
# progress coordinate for the longer task.
EGO_UNIFORM={
 'S':[4,13,17,18,21,25,26,28,29,32,37,39,40,43,44,47,50,51,60,62,64,66,73,76,77],
 'T':[9,49],
 'R':[0,11,12,14,23,34,36,53,54,63,69,70,71,72],
 'U':[19,22,27,56,65],
}
EGO_UNIFORM['P']=sorted(set(range(80))-set().union(*[set(v) for v in EGO_UNIFORM.values()]))

# Whole recordings: S means visible multiple task stages across the six spaced
# frames, not a verified monotone latent trajectory.
EGO_WHOLE={
 'S':[0,1,3,4,5,6,7,9,10,11,13,15,16,17,18,19,22,23],
 'P':[2,8,12,14,20,21],
}

NOTES={
 'kinetics':{
  44:'Apple peel is visibly removed and accumulates.',
  65:'Push-up pose repeats; early pose recurs later.',
  66:'Fuel nozzle and a later gallon display show a measurable state.',
  73:'Pizza toppings accumulate on the dough.',
  80:'Interview cuts to broadcast title card.',
  51:'Selected frames show brick loading and finished pavement, not the laying action.',
 },
 'ego_leaf_uniform':{
  11:'Spoon repeatedly circles a pan; mixture state is unclear.',
  19:'Only 0.5 seconds, nearly identical frames.',
  25:'Celery pieces accumulate on the board.',
  39:'Round batter pieces accumulate on a tray.',
  56:'Label says add firewood, frames show stirring rice.',
 },
 'ego_whole':{
  9:'Cutting and tray preparation precede a later topped baked dish.',
  14:'Computer activity changes, but completion is not visible.',
  17:'Setup, sushi rolling, and later kitchen stage are visible.',
 },
}

def write(kind,labels):
    data=json.load(open(ROOT/f'{kind}_sample.json'))
    rev={i:k for k,ids in labels.items() for i in ids}
    assert len(rev)==len(data)==sum(map(len,labels.values())),(len(rev),len(data),sum(map(len,labels.values())))
    with open(ROOT/f'{kind}_review.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['index','label','category','observation','duration_s','path','contact_sheet'])
        for r in data:
            i=r['index'];w.writerow([i,r['label'],rev[i],NOTES.get(kind,{}).get(i,''),r['duration_s'],r['path'],f'{kind}/sheet_{i//10:02d}.jpg'])
    return dict(Counter(rev.values()))

if __name__=='__main__':
    print('kinetics',write('kinetics',KINETICS))
    print('ego_leaf_uniform',write('ego_leaf_uniform',EGO_UNIFORM))
    print('ego_whole',write('ego_whole',EGO_WHOLE))
