from torch.utils.data import DataLoader

from trajectories import TrajectoryDataset, seq_collate

def data_loader(path):
    dset = TrajectoryDataset(
        path,
        obs_len=8, #args.obs_len,
        pred_len=12, #args.pred_len,
        skip=1, #args.skip,
        delim='\t') #args.delim
        

    loader = DataLoader(
        dset,
        batch_size=10, #args.batch_size,
        shuffle=True,
        num_workers=0, #args.loader_num_workers,
        collate_fn=seq_collate)
    return dset, loader
