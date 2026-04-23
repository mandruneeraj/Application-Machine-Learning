import os, csv, time, argparse, warnings
import os.path as osp


os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

import torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from models.simba import SIMBA
from data.data_loader import BoneageDataset as Dataset
from utils.average_meter import AverageMeter
from utils.metric_average import metric_average


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--relative-age', default=False, action='store_true')
parser.add_argument('--chronological-age', default=False, action='store_true')
parser.add_argument('--gender-multiplier', default=False, action='store_true')
parser.add_argument('--cropped', default=False, action='store_true')
parser.add_argument('--dataset', default='RHPE', type=str, choices=['RSNA','RHPE'])


parser.add_argument('--data-train', default='RHPE_train', type=str)
parser.add_argument('--ann-path-train', default='annotations/RHPE_Boneage_train.csv', type=str)
parser.add_argument('--data-val', default='RHPE_val', type=str)
parser.add_argument('--ann-path-val', default='annotations/RHPE_Boneage_val.csv', type=str)


parser.add_argument('--heatmaps-train', default='data/train/heatmaps', type=str)
parser.add_argument('--rois-path-train', default='annotations/RHPE_anatomical_ROIs_train.json', type=str)
parser.add_argument('--heatmaps-val', default='data/val/heatmaps', type=str)
parser.add_argument('--rois-path-val', default='annotations/RHPE_anatomical_ROIs_val.json', type=str)

parser.add_argument('--trainval', default=False, action='store_true')
parser.add_argument('--save-folder', default='TRAIN/rhpe_run/')
parser.add_argument('--snapshot', default='boneage_bonet_weights.pth')
parser.add_argument('--optim-snapshot', default='boneage_bonet_optim.pth')
parser.add_argument('--eval-first', default=False, action='store_true')


parser.add_argument('-j', '--workers', default=0, type=int)
parser.add_argument('--batch-size', default=1, type=int)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--patience', default=2, type=int)
parser.add_argument('--start-epoch', type=int, default=1)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--log-interval', type=int, default=30)
parser.add_argument('--gpu', type=str, default='-1')
args = parser.parse_args()

print('Args:', vars(args))
torch.manual_seed(args.seed)

device = torch.device('cpu')
if args.gpu != '-1' and torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda')
print(f'Device: {device}')

os.makedirs(args.save_folder, exist_ok=True)
os.makedirs(args.heatmaps_train, exist_ok=True)
os.makedirs(args.heatmaps_val, exist_ok=True)

net = SIMBA(chronological_age=args.chronological_age, gender_multiplier=args.gender_multiplier)
print(f'Params: {sum(p.data.nelement() for p in net.parameters()):,}')

if osp.exists(args.snapshot):
    print(f'Loading weights from {args.snapshot}')
    snap = torch.load(args.snapshot, map_location='cpu')
    w = net.state_dict()
    filtered = {k: v for k, v in snap.items() if k in w and w[k].shape == v.shape}
    w.update(filtered)
    net.load_state_dict(w)

net = net.to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=args.patience,
                                                  cooldown=5, min_lr=1e-5, eps=1e-4)

if osp.exists(args.optim_snapshot):
    optimizer.load_state_dict(torch.load(args.optim_snapshot, map_location='cpu'))

train_tf = transforms.Compose([transforms.Resize((500,500)),
    transforms.RandomAffine(20, translate=(0.2,0.2), scale=(1,1.2)),
    transforms.RandomHorizontalFlip(), transforms.ToTensor()])
val_tf = transforms.Compose([transforms.Resize((500,500)), transforms.ToTensor()])

if args.trainval:
    train_ds = Dataset([args.data_train, args.data_val],
        [args.heatmaps_train, args.heatmaps_val],
        [args.ann_path_train, args.ann_path_val],
        [args.rois_path_train, args.rois_path_val],
        img_transform=train_tf, crop=args.cropped, dataset=args.dataset)
else:
    train_ds = Dataset([args.data_train],[args.heatmaps_train],
        [args.ann_path_train],[args.rois_path_train],
        img_transform=train_tf, crop=args.cropped, dataset=args.dataset)

val_ds = Dataset([args.data_val],[args.heatmaps_val],
    [args.ann_path_val],[args.rois_path_val],
    img_transform=val_tf, crop=args.cropped, dataset=args.dataset)

train_loader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, num_workers=args.workers)
val_loader   = DataLoader(val_ds, shuffle=False, batch_size=1, num_workers=args.workers)


def train_epoch(epoch):
    net.train()
    total = AverageMeter(); epoch_m = AverageMeter(); t_m = AverageMeter()
    optimizer.zero_grad()
    for i, (imgs, ba, gnd, ca, _) in enumerate(train_loader):
        imgs, ba, gnd, ca = imgs.to(device), ba.to(device), gnd.to(device), ca.to(device)
        target = (ca.squeeze(1) - ba) if args.relative_age else ba
        t0 = time.time()
        out = net(imgs, gnd, ca)
        loss = criterion(out.squeeze(), target)
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        v = loss.item()
        t_m.update(time.time()-t0); total.update(v); epoch_m.update(v)
        if i % args.log_interval == 0:
            print(f'  [{epoch:3d}] ({i:4d}/{len(train_loader)}) ms/b={t_m.avg*1000:.1f} loss={total.avg:.4f} lr={optimizer.param_groups[0]["lr"]:.2e}')
            total.reset()
    torch.save(net.state_dict(), osp.join(args.save_folder,'boneage_bonet_snapshot.pth'))
    torch.save(optimizer.state_dict(), osp.join(args.save_folder,'boneage_bonet_optim.pth'))
    return epoch_m.avg


def evaluate():
    net.eval()
    m = AverageMeter()
    with torch.no_grad():
        for imgs, ba, gnd, ca, _ in val_loader:
            imgs, ba, gnd, ca = imgs.to(device), ba.to(device), gnd.to(device), ca.to(device)
            target = (ca.squeeze(1) - ba) if args.relative_age else ba
            out = net(imgs, gnd, ca)
            m.update(criterion(out.squeeze(), target).item())
    print(f'  Val MAE: {m.avg:.4f}')
    return m.avg


def main():
    best = None
    if args.eval_first: evaluate()
    with open(osp.join(args.save_folder,'train_log.csv'),'a') as logf:
        for epoch in range(args.start_epoch, args.epochs+1):
            t0 = time.time()
            tl = train_epoch(epoch)
            scheduler.step(tl)
            vl = evaluate()
            print(f'Epoch {epoch:3d} | time {time.time()-t0:.1f}s | train={tl:.4f} | val={vl:.4f}')
            logf.write(f'{epoch},{tl},{vl}\n'); logf.flush()
            if best is None or vl < best:
                best = vl
                torch.save(net.state_dict(), osp.join(args.save_folder,'boneage_bonet_weights.pth'))
                print(f'  ✅ Best saved (val={best:.4f})')


if __name__ == "__main__":
    main()