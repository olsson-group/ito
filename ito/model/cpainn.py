import warnings

import torch
from torch_scatter import scatter

from ito.model import embedding


class PaiNNTLScore(torch.nn.Module):
    def __init__(
        self,
        n_features=32,
        n_layers=2,
        max_lag=1000,
        diff_steps=1000,
        n_neighbors=100,
        n_types=167,
        dist_encoding="positional_encoding",
    ):
        super().__init__()
        self.embed = torch.nn.Sequential(
            embedding.AddEdges(n_neighbors=n_neighbors),
            embedding.AddEquivariantFeatures(n_features),
            embedding.NominalEmbedding("atom_number", n_features, n_types=n_types),
            embedding.PositionalEmbedding("t_phys", n_features, max_lag),
            embedding.CombineInvariantFeatures(2 * n_features, n_features),
            PaiNNBase(
                n_features=n_features,
                n_features_out=n_features,
                n_layers=n_layers,
                dist_encoding=dist_encoding,
            ),
        )

        self.net = torch.nn.Sequential(
            embedding.AddEdges(should_generate_edge_index=False),
            embedding.PositionalEmbedding("t_diff", n_features, diff_steps),
            embedding.CombineInvariantFeatures(2 * n_features, n_features),
            PaiNNBase(n_features=n_features, dist_encoding=dist_encoding),
        )

    def forward(self, noise_batch, batch_0):
        batch_0 = batch_0.clone()
        noise_batch = noise_batch.clone()

        embedded = self.embed(batch_0)
        cond_inv_features = embedded.invariant_node_features
        cond_eqv_features = embedded.equivariant_node_features
        cond_edge_index = embedded.edge_index

        noise_batch.invariant_node_features = cond_inv_features
        noise_batch.equivariant_node_features = cond_eqv_features
        noise_batch.edge_index = cond_edge_index

        dx = self.net(noise_batch).equivariant_node_features.squeeze()
        noise_batch.x = noise_batch.x + dx

        return noise_batch


class PaiNNBase(torch.nn.Module):
    def __init__(
        self,
        n_features=128,
        n_layers=5,
        n_features_out=1,
        length_scale=10,
        dist_encoding="positional_encoding",
    ):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(
                Message(
                    n_features=n_features,
                    length_scale=length_scale,
                    dist_encoding=dist_encoding,
                )
            )
            layers.append(Update(n_features))

        layers.append(Readout(n_features, n_features_out))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, batch):
        return self.layers(batch)


class Message(torch.nn.Module):
    def __init__(
        self, n_features=128, length_scale=10, dist_encoding="positional_encoding"
    ):
        super().__init__()
        self.n_features = n_features

        assert dist_encoding in (
            a := ["positional_encoding", "soft_one_hot"]
        ), f"positional_encoder must be one of {a}"

        if dist_encoding in ["positional_encoding", None]:
            self.positional_encoder = embedding.PositionalEncoder(
                n_features, length=length_scale
            )
        elif dist_encoding == "soft_one_hot":
            self.positional_encoder = embedding.SoftOneHotEncoder(
                n_features, max_radius=length_scale
            )

        self.phi = embedding.MLP(n_features, n_features, 4 * n_features)
        self.W = embedding.MLP(n_features, n_features, 4 * n_features)

    def forward(self, batch):
        src_node = batch.edge_index[0]
        dst_node = batch.edge_index[1]

        positional_encoding = self.positional_encoder(batch.edge_dist)
        gates, cross_product_gates, scale_edge_dir, scale_features = torch.split(
            self.phi(batch.invariant_node_features[src_node])
            * self.W(positional_encoding),
            self.n_features,
            dim=-1,
        )
        gated_features = multiply_first_dim(
            gates, batch.equivariant_node_features[src_node]
        )
        scaled_edge_dir = multiply_first_dim(
            scale_edge_dir, batch.edge_dir.unsqueeze(1).repeat(1, self.n_features, 1)
        )

        dst_node_edges = batch.edge_dir.unsqueeze(1).repeat(1, self.n_features, 1)
        dst_equivariant_node_features = batch.equivariant_node_features[dst_node]
        cross_produts = torch.cross(
            dst_node_edges, dst_equivariant_node_features, dim=-1
        )

        gated_cross_products = multiply_first_dim(cross_product_gates, cross_produts)

        dv = scaled_edge_dir + gated_features + gated_cross_products
        ds = multiply_first_dim(scale_features, batch.invariant_node_features[src_node])

        dv = scatter(dv, dst_node, dim=0)
        ds = scatter(ds, dst_node, dim=0)

        batch.equivariant_node_features += dv
        batch.invariant_node_features += ds

        return batch


def multiply_first_dim(w, x):
    with warnings.catch_warnings(record=True):
        return (w.T * x.T).T


class Update(torch.nn.Module):
    def __init__(self, n_features=128):
        super().__init__()
        self.U = EquivariantLinear(n_features, n_features)
        self.V = EquivariantLinear(n_features, n_features)
        self.n_features = n_features
        self.mlp = embedding.MLP(2 * n_features, n_features, 3 * n_features)

    def forward(self, batch):
        v = batch.equivariant_node_features
        s = batch.invariant_node_features

        Vv = self.V(v)
        Uv = self.U(v)

        Vv_norm = Vv.norm(dim=-1)
        Vv_squared_norm = Vv_norm**2

        mlp_in = torch.cat([Vv_norm, s], dim=-1)

        gates, scale_squared_norm, add_invariant_features = torch.split(
            self.mlp(mlp_in), self.n_features, dim=-1
        )

        delta_v = multiply_first_dim(Uv, gates)
        delta_s = Vv_squared_norm * scale_squared_norm + add_invariant_features

        batch.invariant_node_features = batch.invariant_node_features + delta_s
        batch.equivariant_node_features = batch.equivariant_node_features + delta_v

        return batch


class EquivariantLinear(torch.nn.Module):
    def __init__(self, n_features_in, n_features_out):
        super().__init__()
        self.linear = torch.nn.Linear(n_features_in, n_features_out, bias=False)

    def forward(self, x):
        return self.linear(x.swapaxes(-1, -2)).swapaxes(-1, -2)


class Readout(torch.nn.Module):
    def __init__(self, n_features=128, n_features_out=13):
        super().__init__()
        self.mlp = embedding.MLP(n_features, n_features, 2 * n_features_out)
        self.V = EquivariantLinear(n_features, n_features_out)
        self.n_features_out = n_features_out

    def forward(self, batch):
        invariant_node_features_out, gates = torch.split(
            self.mlp(batch.invariant_node_features), self.n_features_out, dim=-1
        )

        equivariant_node_features = self.V(batch.equivariant_node_features)
        equivariant_node_features_out = multiply_first_dim(
            equivariant_node_features, gates
        )

        batch.invariant_node_features = invariant_node_features_out
        batch.equivariant_node_features = equivariant_node_features_out
        return batch
