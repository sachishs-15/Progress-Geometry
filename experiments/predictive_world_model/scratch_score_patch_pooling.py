import sys, torch, yaml, inspect, numpy as np, importlib.util
sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location("cpm", "../encoder_planning_eval/progress_metrics/compute_progress_metrics.py")
cpm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpm)
sys.path.insert(0, '../../external/chirality_in_action/LiFT/visualization')
from progress_metrics import clip_progress_metrics
from models.world_model import ProceduralWorldModel

sig = set(inspect.signature(ProceduralWorldModel.__init__).parameters) - {'self'}

patch_x = torch.load('/mnt/rdata7/proprogress/data/SSv2/precomputed_features/dinov2_noreg/ssv2_validation.pt',
                      map_location='cpu')['features'].float()


def score_mean(z):
    z = z.astype(np.float64)
    per = []
    for i in range(len(z)):
        c = z[i]
        m = clip_progress_metrics(c)
        if m is None:
            continue
        per.append(dict(spearman=m['time_progress_spearman'], straight=m['trajectory_straightness'],
                         disp=m['start_end_displacement'], d2g=cpm.distance_to_goal_spearman(c),
                         velcos=cpm.mean_velocity_cosine(c)))
    o = {k: float(np.mean([p[k] for p in per if p[k] is not None])) for k in per[0]}
    flat = z.reshape(-1, z.shape[-1])
    if flat.shape[1] > 5000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(flat), size=20000, replace=False)
        flat = flat[idx]
    o['effrank'] = cpm.effective_rank(flat)[0]
    return o


def run_patch_batched(model, feats, bs=32, device='cuda:0'):
    model = model.to(device).eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(feats), bs):
            x = feats[i:i + bs].to(device).float()
            z = model.encoder(x)
            chunks.append(z.cpu())
    return torch.cat(chunks, 0)


def pool(z, mode):
    if mode == 'mean':
        return z.mean(dim=2).numpy()
    if mode == 'concat':
        return z.reshape(z.shape[0], z.shape[1], -1).numpy()


cols = ['spearman', 'straight', 'disp', 'd2g', 'velcos', 'effrank']
hdr = ['spearmn', 'straight', 'disp', 'd2goal', 'velcos', 'effrnk']
print(f'{"":<28s} {"pool":<7s}' + ''.join(f'{h:>9s}' for h in hdr), flush=True)


def row(l, mode, s):
    print(f'{l:<28s} {mode:<7s}' + ''.join(f'{s[c]:9.4f}' for c in cols), flush=True)


for mode in ['mean', 'concat']:
    row('raw dinov2 patch', mode, score_mean(pool(patch_x, mode)))

runs = [('patch none SSv2', 'patch_channel_paper_window0_ssv2'),
        ('patch vicreg SSv2', 'patch_channel_paper_window0_vicreg_ssv2_1'),
        ('patch sigreg SSv2', 'patch_channel_paper_window0_sigreg_ssv2')]
first = True
for lbl, ck in runs:
    p = 'checkpoints/pwm_' + ck
    cfg = yaml.safe_load(open(p + '/config.yaml'))
    arch = {k: v for k, v in cfg.items() if k in sig}
    if first:
        torch.manual_seed(0)
        mu = ProceduralWorldModel(**arch)
        zu = run_patch_batched(mu, patch_x)
        for mode in ['mean', 'concat']:
            row('UNTRAINED patch', mode, score_mean(pool(zu, mode)))
        first = False
    m = ProceduralWorldModel(**arch)
    m.load_state_dict(torch.load(p + '/last.ckpt', map_location='cpu')['model'], strict=False)
    z = run_patch_batched(m, patch_x)
    for mode in ['mean', 'concat']:
        row(lbl, mode, score_mean(pool(z, mode)))

print("DONE", flush=True)
