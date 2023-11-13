import json
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from deeptime import decomposition

from ito import data, utils


def main(args):
    args.root = os.path.realpath(args.root)
    topology = data.get_ala2_top(args.root)
    analysis_dir = os.path.join(args.root, "analysis", utils.get_timestamp())
    os.makedirs(analysis_dir, exist_ok=True)

    json.dump(
        args.__dict__, open(os.path.join(analysis_dir, "args.json"), "w"), indent=4
    )

    trajs = np.load(args.trajs)
    vamp2_score = get_vamp2(trajs=trajs, topology=topology, lag=1)

    ref_trajs = data.get_ala2_trajs(args.root)
    ref_vamp2_score = get_vamp2(ref_trajs, topology=topology, lag=args.lag)
    json.dump(
        {"vamp2": vamp2_score, "ref_vamp2": ref_vamp2_score},
        open(os.path.join(analysis_dir, "vamp2_scores.json"), "w"),
        indent=4,
    )

    print(f"VAMP2 score: {vamp2_score}")
    print(f"Reference VAMP2 score: {ref_vamp2_score}")

    phi, psi = compute_dihedral_angles(np.concatenate(trajs), topology)
    phi_ref, psi_ref = compute_dihedral_angles(np.concatenate(ref_trajs), topology)

    _, ax = plt.subplots(1, 2, figsize=(10, 4))

    plot_marginal(ax[0], phi)
    plot_marginal(ax[0], phi_ref, label="MD")
    ax[0].set_xlabel("Phi")
    ax[0].set_ylabel("Frequency")
    if not args.no_plot_start:
        ax[0].vlines(phi[0], 0, 1, linestyle="--", color="k")

    plot_marginal(ax[1], psi, label="ITO")
    plot_marginal(ax[1], psi_ref, label="MD")
    ax[1].set_xlabel("Psi")
    if not args.no_plot_start:
        ax[1].vlines(psi[0], 0, 1, linestyle="--", color="k", label="Start")

    ax[1].legend()
    plt.savefig(os.path.join(analysis_dir, "marginals.pdf"))

    _, ax = plt.subplots(figsize=(5, 4))
    plt.plot(phi, psi, ".", alpha=0.1, label="ITO")
    ax.set_xlabel("Phi")
    ax.set_ylabel("Psi")
    if not args.no_plot_start:
        plt.plot(phi[0], psi[0], "k", marker="x", label="Start")

    ax.legend()
    plt.savefig(os.path.join(analysis_dir, "ramachandran.pdf"))
    plt.show()


def plot_marginal(ax, marginal, bins=64, label=None):
    bin_heights, _ = np.histogram(
        marginal, range=(-np.pi, np.pi), bins=bins, density=True
    )
    plot_marginal_dist(ax, bin_heights, label=label, linestyle="-")


def plot_marginal_dist(ax, bin_heights, linestyle="-", c=None, label=None):
    bin_edges = np.linspace(-np.pi, np.pi, len(bin_heights) + 1)
    bin_widths = np.diff(bin_edges)
    bin_heights /= bin_heights.mean() * bin_widths.sum()
    bin_heights = np.append(bin_heights, bin_heights[-1])

    ax.step(
        bin_edges,
        bin_heights,
        where="post",
        linestyle=linestyle,
        c=c,
        label=label,
    )
    ax.semilogy()


def compute_dihedral_angles(traj, topology):
    traj = md.Trajectory(xyz=traj, topology=topology)
    phi_atoms_idx = [4, 6, 8, 14]
    phi = md.compute_dihedrals(traj, indices=[phi_atoms_idx])[:, 0]
    psi_atoms_idx = [6, 8, 14, 16]
    psi = md.compute_dihedrals(traj, indices=[psi_atoms_idx])[:, 0]

    return phi, psi


def featurize_trajs(trajs, topology):
    featurized_trajs = np.stack(featurize_traj(traj, topology) for traj in trajs)
    nan_mask = np.isnan(featurized_trajs).any(axis=(1, 2))

    return featurized_trajs[~nan_mask]


def featurize_traj(traj, topology):
    phi, psi = compute_dihedral_angles(traj, topology)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    features = np.stack([cos_phi, sin_phi, cos_psi, sin_psi]).T
    return features


def get_vamp2(trajs, lag, topology):
    featurized_trajs = featurize_trajs(trajs, topology)
    vamp = decomposition.VAMP(lag).fit_fetch(featurized_trajs)
    vamp2_score = vamp.score(2)

    return vamp2_score


if __name__ == "__main__":
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument( "--trajs", default="storage/samples/latest", help="Specify the path to the trajectory file containing the trajectories to be analyzed. Default is 'storage/samples/latest'. If the default path is unchanged, ensure that the sampling script has been run prior to analysis.")
    parser.add_argument( "--root",  default="storage", help="Set the base directory where input data is located and where analysis outputs will be stored. The default directory is 'storage'. Modify this if your data and output directories are different.")
    parser.add_argument( "--lag",   type=int, default=100, help="Define the temporal lag (in steps) between frames in the ITO trajectory. This value will be used to analyse the reference trajs such that time steps match. The default value is 100.")
    parser.add_argument( "--no_plot_start", action="store_false", help="Include this flag to prevent marking the starting point of trajectories in the generated plots. By default, the starting point is marked. This should only be used if the trajectories was generated with --init_from_eq")
    # fmt: on

    main(parser.parse_args())
