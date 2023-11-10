import datetime

import torch
from torch_geometric.data import Batch, Data


def get_batch_from_atom_numbers_and_position(atom_numbers, positions):
    if len(positions.shape) == 2: # single frame
        positions = positions[None,:]


    datalist = [
        Data(
            x=torch.tensor(positions[i]),  # pylint: disable=not-callable
            atom_number=atom_numbers,  # pylint: disable=not-callable
        )
        for i in range(len(positions))
    ]
    batch = Batch.from_data_list(datalist)
    return batch


def add_t_phys_to_batch(batch, t_phys):
    batch.t_phys = torch.ones_like(batch.atom_number) * t_phys
    return batch


def get_cond_batch(atom_numbers, positions, t_phys):
    batch = get_batch_from_atom_numbers_and_position(atom_numbers, positions)
    batch = add_t_phys_to_batch(batch, t_phys)
    return batch


def batch_to_tensor(batch):
    poss = [data.x for data in batch.to_data_list()]
    return torch.stack(poss)


def batch_to_numpy(batch):
    return batch_to_tensor(batch).cpu().numpy()


def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
