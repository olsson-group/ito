import json
from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from ito import data
from ito.model import cpainn, ddpm


def main(args):

    score_model = cpainn.PaiNNTLScore(
        n_features=args.n_features, max_lag=args.max_lag, diff_steps=args.diff_steps
    )

    model = ddpm.TLDDPM(
        score_model=score_model, diffusion_steps=args.diff_steps, lr=args.lr
    )

    dataset = data.ALA2Dataset(
        path="data/ala2",
        max_lag=args.max_lag,
        distinguish=args.distinguish,
        fixed_lag=args.fixed_lag,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, dataloader)




if __name__ == "__main__":
    parser = ArgumentParser()

    # fmt: off
    parser.add_argument("--n_features",  nargs="?", type=int,   default=64)
    parser.add_argument("--n_layers",    nargs="?", type=int,   default=2)
    parser.add_argument("--epochs",      nargs="?", type=int,   default=50)
    parser.add_argument("--diff_steps",  nargs="?", type=int,   default=1000)
    parser.add_argument("--batch_size",  nargs="?", type=int,   default=128)
    parser.add_argument("--lr",          nargs="?", type=float, default=1e-3)
    parser.add_argument("--max_lag",     nargs="?", type=int,   default=1000)
    parser.add_argument("--fixed_lag",   action='store_true')
    parser.add_argument("--distinguish", action='store_true')
    parser.add_argument("--scale",       action='store_true')
    # fmt: on

    main(parser.parse_args())

    main(parser.parse_args())
