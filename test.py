import os, argparse, warnings
import os.path as osp
import pandas as pd 
import torch, torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.simba import SIMBA
from data.data_loader import BoneageDataset as Dataset
from utils.average_meter import AverageMeter
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--relative-age', default=False, action='store_true')
parser.add_argument('--chronological-age', default=False, action='store_true')
parser.add_argument('--gender-multiplier', default=False, action='store_true')
parser.add_argument('--cropped', default=False, action='store_true')
parser.add_argument('--dataset', default='RHPE', type=str, choices=['RSNA','RHPE'])
parser.add_argument('--data-test', default='data/test/images', type=str)
parser.add_argument('--heatmaps-test', default='data/test/heatmaps', type=str)
parser.add_argument('--ann-path-test', default='annotations/test_annotations.csv', type=str)
parser.add_argument('--rois-path-test', default='annotations/RHPE_anatomical_ROIs_test.json', type=str)
parser.add_argument('--save-folder', default='TRAIN/rhpe_run/')
parser.add_argument('--snapshot', default='TRAIN/rhpe_run/boneage_bonet_weights.pth')
parser.add_argument('--save-file', default='predictions.csv')
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--batch-size', default=1, type=int)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--inference-only', default=False, action='store_true',
                help='Use when test CSV has no ground-truth bone age')
args = parser.parse_args()

print('Args:', vars(args))
torch.manual_seed(args.seed)

device = torch.device('cpu')
if args.gpu != '-1' and torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda')
print(f'Device: {device}')

os.makedirs(osp.join(args.save_folder,'inference'), exist_ok=True)
os.makedirs(args.heatmaps_test, exist_ok=True)

net = SIMBA(chronological_age=args.chronological_age, gender_multiplier=args.gender_multiplier)
print(f'Params: {sum(p.data.nelement() for p in net.parameters()):,}')

if osp.exists(args.snapshot):
    print(f'Loading weights from {args.snapshot}')
    snap = torch.load(args.snapshot, map_location='cpu')
    w = net.state_dict()
    filtered = {k: v for k, v in snap.items() if k in w and w[k].shape == v.shape}
    w.update(filtered)
    net.load_state_dict(w)
else:
    print(f'WARNING: No snapshot at {args.snapshot} — using random weights')

net = net.to(device)
criterion = nn.L1Loss()

test_tf = transforms.Compose([transforms.Resize((500,500)), transforms.ToTensor()])

test_ds = Dataset([args.data_test],[args.heatmaps_test],
    [args.ann_path_test],[args.rois_path_test],
    img_transform=test_tf, crop=args.cropped, dataset=args.dataset,
    inference=args.inference_only)

test_loader = DataLoader(test_ds, shuffle=False, batch_size=1, num_workers=0)


def main():
    print('Inference begins...')
    df = pd.read_csv(args.ann_path_test)
    p_dict = dict.fromkeys(df['ID'].tolist())

    net.eval()
    for child in net.children():
        if type(child) == nn.BatchNorm2d:
            child.track_running_stats = False

    loss_m = AverageMeter()
    results = {}

    with torch.no_grad():
        for imgs, ba, gnd, ca, pid in tqdm(test_loader, total=len(test_loader)):
            imgs, ba, gnd, ca = imgs.to(device), ba.to(device), gnd.to(device), ca.to(device)
            out = net(imgs, gnd, ca)
            pred = out.item()
            sample_id = pid.item()
            results[sample_id] = pred

            if not args.inference_only:
                target = (ca.squeeze(1) - ba) if args.relative_age else ba
                loss_m.update(criterion(out.squeeze(), target).item())

    if not args.inference_only:
        print(f'Test MAE: {loss_m.avg:.4f} months')

    out_df = pd.DataFrame(list(results.items()), columns=['ID', 'predicted_bone_age'])
    out_path = osp.join(args.save_folder, 'inference', args.save_file)
    out_df.to_csv(out_path, index=False)
    print(f'Predictions saved to: {out_path}')
    print(out_df.head(10).to_string())


if __name__ == '__main__':
    main()