import json
import os
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from ito import data, utils
from ito.model import ddpm


def main(args):
    args.root = os.path.realpath(args.root)
    ala2_path = os.path.join(args.root, "data/ala2")
    samples_root = os.path.join(args.root, "samples")
    samples_dir = os.path.join(samples_root, utils.get_timestamp())
    samples_path = os.path.join(samples_dir, "trajectory.npy")

    ala2_trajs = np.concatenate(data.get_ala2_trajs(ala2_path, not args.unscaled))
    np.random.shuffle(ala2_trajs)
    ala2_atom_numbers = data.get_ala2_atom_numbers(not args.indistinguishable)

    model = ddpm.TLDDPM.load_from_checkpoint(args.checkpoint)

    if args.init_from_eq:
        init_positions = ala2_trajs[: args.samples]
    else:
        init_positions = ala2_trajs[None, 0].repeat(args.samples, axis=0)

    batch = utils.get_cond_batch(ala2_atom_numbers, init_positions, args.lag)

    trajectory = [utils.batch_to_numpy(batch)]

    for _ in tqdm(range(args.traj_length)):
        batch = model.sample(batch, ode_steps=args.ode_steps)
        trajectory.append(utils.batch_to_numpy(batch))

    trajectory = np.stack(trajectory, axis=1)

    os.makedirs(samples_dir, exist_ok=True)
    json.dump(
        args.__dict__, open(os.path.join(samples_dir, "args.json"), "w"), indent=4
    )
    np.save(samples_path, trajectory)
    os.symlink(src=os.path.abspath(samples_path), dst=os.path.join(samples_root, "latest"))

    print(f"samples saved at {samples_path}")



if __name__ == "__main__":
    parser = ArgumentParser()

    # fmt: off
    parser.add_argument("checkpoint",          nargs="?",           default="storage/train/latest/best", help="Path to the model checkpoint file for generating trajectories. Default is the best model from the latest training session.")
    parser.add_argument("--root",              default="storage",   help="Base path for input data and where output samples will be stored.")
    parser.add_argument("--samples",           type=int,            default=10,                          help="The number of initial positions to sample trajectories from, processed in parallel.")
    parser.add_argument("--traj_length",       type=int,            default=100,                         help="The total number of steps (frames) in each generated trajectory.")
    parser.add_argument("--lag",               type=int,            default=100,                         help="Temporal lag between consecutive steps (frames) in the generated trajectory.")
    parser.add_argument("--ode_steps",         type=int,            default=50,                          help="Number of steps for the ODE solver during sampling. Set to 0 for normal denoising.")
    parser.add_argument("--indistinguishable", action="store_true", help="Enable this flag to treat atoms as indistinguishable in the model.")
    parser.add_argument("--unscaled",          action="store_true", help="Enable this flag to use unscaled data. By default, data is scaled to have unit variance.")
    parser.add_argument("--init_from_eq",           action="store_true", help="Initiate trajectories at from random configurations from the equilibrium distributions rather than from a single point")
    # fmt: on

    main(parser.parse_args())
