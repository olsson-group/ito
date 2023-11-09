# # pylint: disable=not-callable
import sys

import numpy as np
import torch
#  import torch_cluster
import torch_geometric
#  from e3nn.math import soft_one_hot_linspace
#
#  from dynamic_diffusion import DEVICE
#
#
class MLP(torch.nn.Module):
    def __init__(self, f_in, f_hidden, f_out, skip_connection=False):
        super().__init__()
        self.skip_connection = skip_connection

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(f_in, f_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(f_hidden, f_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(f_hidden, f_out),
        )

    def forward(self, x):
        if self.skip_connection:
            return x + self.mlp(x)

        return self.mlp(x)


#  class Clone(torch.nn.Module):
#      @staticmethod
#      def forward(batch):
#          return batch.clone()
#
#
class AddEdges(torch.nn.Module):
    def __init__(self, n_neighbors=None, cutoff=None, should_generate_edge_index=True):
        super().__init__()
        self.n_neighbors = n_neighbors if n_neighbors else sys.maxsize
        self.cutoff = cutoff if cutoff is not None else float("inf")
        self.should_generate_edge_index = should_generate_edge_index

    def forward(self, batch, *args, **kwargs):
        batch = self.add_edges(batch)
        return batch

    def add_edges(self, batch):
        if self.should_generate_edge_index:
            self.generate_edge_index(batch)

        if hasattr(batch, "is_fictitious"):
            self.connect_fictitious_atom(batch)

        r = batch.x[batch.edge_index[0]] - batch.x[batch.edge_index[1]]
        edge_dist = r.norm(dim=-1)
        edge_dir = r / (1 + edge_dist.unsqueeze(-1))

        batch.edge_dist = edge_dist
        batch.edge_dir = edge_dir
        return batch

    @staticmethod
    def connect_fictitious_atom(batch):
        fictitious_atoms = torch.where(batch.is_fictitious == 1)[0]
        real_atoms = torch.where(batch.is_fictitious == 0)[0]
        real_atoms_batch = batch.batch[real_atoms]

        # one day you'll find an error here.
        src = torch.cat(
            (fictitious_atoms[real_atoms_batch], real_atoms, fictitious_atoms)
        )
        dst = torch.cat(
            (real_atoms, fictitious_atoms[real_atoms_batch], fictitious_atoms)
        )

        edge_index = torch.stack((src, dst))

        batch.edge_index = torch.concatenate((batch.edge_index, edge_index), dim=1)
        batch.edge_index = torch_geometric.utils.coalesce(batch.edge_index)

    def generate_edge_index(self, batch):
        edge_index = torch_geometric.nn.radius_graph(
            batch.x,
            r=self.cutoff,
            batch=batch.batch,
            max_num_neighbors=self.n_neighbors,
        )
        batch.edge_index = edge_index
#
#
#  def calc_direction(edge_index, batch, dist=None):
#      if dist is None:
#          dist = calc_dist(edge_index, batch)
#      return (batch.x[edge_index[0]] - batch.x[edge_index[1]]) / (1 + dist.unsqueeze(-1))
#
#
#  def calc_dist(edge_index, batch):
#      return (batch.x[edge_index[0]] - batch.x[edge_index[1]]).norm(dim=-1)
#
#
class InvariantFeatures(torch.nn.Module):
    """
    Implement embedding in child class
    All features that will be embedded should be in the batch
    """

    def __init__(self, feature_name):
        super().__init__()
        self.feature_name = feature_name

    def forward(self, batch):
        embedded_features = self.embedding(batch[self.feature_name])

        if hasattr(batch, "invariant_node_features"):
            batch.invariant_node_features = torch.cat(
                [batch.invariant_node_features, embedded_features], dim=-1
            )
        else:
            batch.invariant_node_features = embedded_features

        return batch
#
#
#  class AddInvariantFeatures(InvariantFeatures):
#      def __init__(self, feature_name):  # pylint: disable=useless-super-delegation
#          super().__init__(feature_name)
#          self.embedding = torch.nn.Identity()
#
#
class NominalEmbedding(InvariantFeatures):
    def __init__(self, feature_name, n_features, n_types):
        super().__init__(feature_name)
        self.embedding = torch.nn.Embedding(n_types, n_features)


class PositionalEncoder(torch.nn.Module):
    def __init__(self, dim, length=10):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for positional encoding for sin/cos"
        self.dim = dim
        self.length = length
        self.max_rank = dim // 2

    def forward(self, x):
        return torch.concatenate(
            [self.positional_encoding(x, rank) for rank in range(self.max_rank)],
            axis=1,
        )

    def positional_encoding(self, x, rank):
        sin = torch.sin(x / self.length * rank * np.pi)
        cos = torch.cos(x / self.length * rank * np.pi)
        return torch.stack((cos, sin), axis=1)


class PositionalEmbedding(InvariantFeatures):
    def __init__(self, feature_name, n_features, length):
        super().__init__(feature_name)
        assert n_features % 2 == 0, "n_features must be even"
        self.rank = n_features // 2
        self.embedding = PositionalEncoder(n_features, length)


#  class SoftOneHotEncoder(torch.nn.Module):
#      def __init__(self, n_rbf, max_radius, basis="cosine"):
#          super().__init__()
#          self.n_rbf = n_rbf
#          self.max_radius = max_radius
#          self.basis = basis
#
#      def forward(self, r):
#          return soft_one_hot_linspace(
#              r,
#              0.0,
#              self.max_radius,
#              self.n_rbf,
#              basis=self.basis,
#              cutoff=True,
#          ).mul(self.n_rbf**0.5)
#
#
#  class SoftOneHotEmbedding(InvariantFeatures):
#      def __init__(self, feature_name, n_features, max_radius, basis="cosine"):
#          super().__init__(feature_name)
#          self.embedding = SoftOneHotEncoder(
#              n_rbf=n_features, max_radius=max_radius, basis=basis
#          )
#
#
class CombineInvariantFeatures(torch.nn.Module):
    def __init__(self, n_features_in, n_features_out):
        super().__init__()
        self.mlp = MLP(n_features_in, n_features_out, n_features_out)

    def forward(self, batch):
        batch.invariant_node_features = self.mlp(batch.invariant_node_features)
        return batch


class AddEquivariantFeatures(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.register_buffer("device_tracker", torch.tensor(1))

    @property
    def device(self):
        return self.device_tracker.device

    def forward(self, batch):
        batch.equivariant_node_features = torch.zeros(
            batch.batch.shape[0],
            self.n_features,
            3,
            dtype=torch.float32,
            device=self.device,
        )
        return batch
