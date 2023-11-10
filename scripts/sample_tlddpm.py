import json
import os
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm

from ito import data
from ito.model import cpainn, ddpm
from torch_geometric.data import Batch, Data
from tqdm import tqdm


def main(args):
    path = os.path.join(args.path, "data/ala2")
    ala2_trajs = np.concatenate(data.get_ala2_trajs(path, not args.unscaled))
    np.random.shuffle(ala2_trajs)
    ala2_atom_numbers = data.get_ala2_atom_numbers(not args.indistinguishable)

    model = ddpm.TLDDPM.load_from_checkpoint(args.checkpoint)

    init_positions = ala2_trajs[:args.samples]
    batch = get_init_batch(ala2_atom_numbers, init_positions, args.lag)

    trajectory = [batch.x.detach().numpy()]

    for _ in tqdm(range(args.steps)):
        batch = model.sample(batch, ode_steps=args.ode_steps)
        __import__("pdb").set_trace() #TODO delme 



def get_batch_from_atom_numbers_and_position(atom_numbers, positions):
    datalist = [
        Data(
            x=torch.tensor(positions[i]),  # pylint: disable=not-callable
            atom_number=torch.tensor(atom_numbers),  # pylint: disable=not-callable
        )
        for i in range(len(positions))
    ]
    batch = Batch.from_data_list(datalist)
    return batch


def add_t_phys_to_batch(batch, t_phys):
    batch.t_phys = torch.ones_like(batch.atom_number) * t_phys
    return batch

def get_init_batch(atom_numbers, positions, t_phys):
    batch = get_batch_from_atom_numbers_and_position(atom_numbers, positions)
    batch = add_t_phys_to_batch(batch, t_phys)
    return batch





if __name__ == "__main__":
    # fmt: off
    parser                                     =                    ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--path",              nargs="?",           type=str, default=".", help="path to data")
    parser.add_argument("--samples",           nargs="?",           type=int, default=10,  help="number of samples")
    parser.add_argument("--steps",             nargs="?",           type=int, default=100, help="number of steps")
    parser.add_argument("--lag",               nargs="?",           type=int, default=100, help="lag time")
    parser.add_argument("--ode_steps",         nargs="?",           type=int, default=50,  help="How many ODE-steps to use for sampling, if set to 0 we will denoise normally.")
    parser.add_argument("--indistinguishable", action="store_true", help="Use indistinguishable atoms")
    parser.add_argument("--unscaled",          action="store_true", help="Use unscaled data, otherwise data will be scaled to have a mean variance of 1")
    
    # fmt: on

    # parser.add_argument('--kwarg')

    main(parser.parse_args())


#  import json
#  import multiprocessing as mp
#  import os
#  import pickle as pkl
#  from argparse import ArgumentParser
#
#  import h5py
#  import mdtraj as md
#  import numpy as np
#  import torch
#  import torch.multiprocessing as mp
#  from torch_geometric.data import Batch, Data, DataLoader
#  from tqdm import tqdm
#
#  import wandb
#  from dynamic_diffusion import DEVICE, analysis, data, utils
#  from dynamic_diffusion.model.ddpm import DDPM, TLDDPM
#  from dynamic_diffusion.utils import init_wandb
#
#  try:
#      mp.set_start_method("spawn")
#  except RuntimeError:
#      pass
#
#
#  def get_samples(trajs, n_samples):
#      trajs = np.concatenate(trajs)
#      np.random.shuffle(trajs)
#      samples = trajs[:n_samples]
#      return samples
#
#
#  def get_trajs(system):
#      if system == "cln025":
#          trajs = data.get_cln025_trajs()
#          top = data.get_cln025_topology()
#          calpha_idx = top.select_atom_indices("alpha")
#          trajs = [traj[:, calpha_idx, :] for traj in trajs]
#
#      elif system == "ala2":
#          trajs = data.get_ala2_trajs()
#
#      else:
#          raise ValueError(f"No system {system}")
#
#      return trajs
#
#
#  def main(args):  # pylint: disable=too-many-branches
#      init_wandb("sampling", args.wboff)
#      wandb.log(vars(args))
#      model = get_model(args.method, args.model, args.tag)
#      model.to(DEVICE)
#
#      if args.method == "ddpm":
#          get_sample_kwargs = get_ddpm_sample_kwargs
#
#      elif args.method == "tlddpm":
#          get_sample_kwargs = get_tlddpm_sample_kwargs
#
#      sample_kwargs = get_sample_kwargs(args)
#
#      experiment = os.path.join(args.experiment)
#      results_path = utils.get_results_path("sample", experiment=experiment)
#
#      with open(os.path.join(results_path, "args.json"), "w") as f:
#          json.dump(vars(args), f)
#
#      if args.cond:
#          # logic for building a batch from a condition
#          ...
#
#      else:
#          trajs = get_trajs(args.system)
#          samples = get_samples(trajs, args.n_samples)
#          batch = create_cond_batch_from_array(
#              samples, get_atom_numberss("cln025", distinguish=True), args.lag
#          )
#
#      samples = []
#      chunk_samples = []
#
#      for chunk in tqdm(range(args.chunks), desc="Chunks"):
#          wandb.log({"chunk": chunk})
#          sample_kwargs["cond_batch"] = batch
#
#          np_sample = utils.geomdata_to_numpy(batch)
#          samples = [np_sample]
#
#          for step in tqdm(range(args.nested), desc="Nested steps", leave=False):
#              wandb.log({"step": step})
#              sample = model.sample(**sample_kwargs)
#
#              if args.nested > 1 and args.method == "tlddpm":
#                  nan_mask = torch.isnan(sample.x)
#                  sample_kwargs["cond_batch"].x[~nan_mask] = sample.x[~nan_mask]
#
#              np_sample = utils.geomdata_to_numpy(sample)
#              if args.only_save_last:
#                  samples = [np_sample]
#              else:
#                  samples.append(np_sample)
#
#      samples = np.concatenate(samples, axis=0)
#      samples = group_trajs(samples, args.n_samples)
#      samples = samples * get_scaling_factor(args.system)
#      sample_path = os.path.join(results_path, "samples.npy")
#
#      with open(sample_path, "wb") as f:
#          np.save(f, samples)
#
#      #  for i, sample in enumerate(samples[0]):
#      #      sample *= 10
#      #      utils.vis_mol(sample, data.ALA2ATOMNUMBERS)
#
#
#  def get_scaling_factor(system):
#      if system == "ala2":
#          return 0.1661689
#      if system == "cln025":
#          return 0.37521568
#
#      else:
#          raise ValueError(f"Unkown system {system}")
#
#
#  def update_pos(cond_batch, sample):
#      cond_batch = cond_batch.clone()
#      cond_batch.x = (
#          torch.from_numpy(sample)
#          .float()
#          .to(cond_batch.x.device)
#          .reshape(cond_batch.x.shape)
#      )
#      return cond_batch
#
#
#  def get_ddpm_sample_kwargs(args):
#      atoms = get_atom_numberss(args.system, args.add_fictitious_atom, args.distinguish)
#      sample_kwargs = {
#          "n_samples": args.n_samples,
#          "atom_numbers": atoms,
#          "ode_steps": args.ode_steps,
#      }
#
#      return sample_kwargs
#
#
#  def get_tlddpm_sample_kwargs(args):
#      sample_kwargs = {"ode_steps": args.ode_steps}
#      return sample_kwargs
#
#
#  def get_default_cond(system):
#      system = system.upper()
#      if system in data.PROTEINS:
#          return f"data/proteins/{system}/sample.pkl"
#      if system == "ALA2":
#          return "data/ala2_init_states/0.pkl"
#      raise ValueError(f"Unknown system {system}")
#
#
#  def load_cond_pos(path, n_samples):
#      cond_pos = pkl.load(open(path, "rb"))
#      return cond_pos[None, 0].repeat(n_samples, 0)
#
#
#  def create_cond_batch_from_array(cond_pos, atoms, lag):
#      cond_batch = utils.get_batch_from_atom_numbers_and_position(atoms, cond_pos)
#      add_lag(cond_batch, lag)
#      return cond_batch
#
#
#  def get_model(method, model, tag):
#      MODELS = {"tlddpm": TLDDPM, "ddpm": DDPM}
#      ckpt = utils.get_checkpoint(method, model, tag)
#      model_cls = MODELS[method]
#      model = model_cls.load_from_checkpoint(ckpt, strict=False)
#      return model
#
#
#  def get_atom_numberss(system, distinguish):
#      system = system.upper()
#      if system == "ALA2":
#          atoms = data.ALA2ATOMNUMBERS
#
#      elif system in data.RESNUMBERS:
#          atoms = data.RESNUMBERS[system]
#
#      else:
#          raise ValueError(f"Unknown system {system}")
#
#      if distinguish:
#          atoms = list(range(len(atoms)))
#
#      return atoms
#
#
#  def add_lag(cond_batch, lag):
#      cond_batch.t_phys = torch.ones_like(cond_batch.atom_numbers) * lag
#
#
#  class BatchGetter:
#      def __init__(self, lag, n_samples, system, distinguish, random=False, idx=0):
#          self.random = random
#          self.idx = idx
#          self.ds = data.GeomH5ProteinDataset(proteins=[system], distinguish=distinguish)
#          self.dl = DataLoader(self.ds, batch_size=n_samples, shuffle=True)
#          self.n_samples = n_samples
#          self.lag = lag
#
#      def get_random_batch(
#          self,
#      ):
#          cond_batch = next(iter(self.dl))
#          add_lag(cond_batch, self.lag)
#          return cond_batch
#
#      def get_batch_from_idx(self, idx):
#          batch = self.ds[idx]
#          add_lag(batch, self.lag)
#          data_list = [batch for _ in range(self.n_samples)]
#          return Batch.from_data_list(data_list)
#
#      def get_batch(self):
#          if self.random:
#              return self.get_random_batch()
#          return self.get_batch_from_idx(self.idx)
#
#
#  def group_trajs(trajs, n_samples):
#      trajs = trajs.reshape(-1, n_samples, *trajs.shape[1:])
#      return np.transpose(trajs, (1, 0, *range(2, len(trajs.shape))))
#
#
#  if __name__ == "__main__":
#      parser = ArgumentParser()
#      parser.add_argument("method")
#      parser.add_argument("system")
#      parser.add_argument("model", nargs="?")
#      parser.add_argument("tag", nargs="?", default="best")
#      parser.add_argument("--molecule", type=str, default=None)
#      parser.add_argument("--experiment", type=str, default="unversioned")
#      parser.add_argument("--cond", type=str, default=None)
#      parser.add_argument("--lag", type=int, default=1000)
#      parser.add_argument("--n_samples", type=int, default=1)
#      parser.add_argument("--ode_steps", type=int, default=0)
#      parser.add_argument("--atoms", type=str, default=None)
#      parser.add_argument("--chunks", type=int, default=1)
#      parser.add_argument("--nested", type=int, default=1)
#      parser.add_argument("--distinguish", action="store_true")
#      parser.add_argument("--wboff", action="store_true")
#      parser.add_argument("--add_fictitious_atom", action="store_true")
#      parser.add_argument("--scale", action="store_true")
#      parser.add_argument("--only_save_last", action="store_true")
#      parser.add_argument("--dont_save_init", action="store_true")
#      parser.add_argument("--random_ds", action="store_true")
#      parser.add_argument("--idx_ds", type=int, default=0)
#
#      main(parser.parse_args())
#
