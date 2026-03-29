from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        embed_dim=16,
        num_heads=2,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.embedding = torch.nn.Embedding(n_track, embed_dim)

        # to transform last dim to embed_dim
        # 2 * 2 because concating left and right track
        self.projection = torch.nn.Linear(2 * 2, embed_dim)

        self.in_norm = torch.nn.LayerNorm(normalized_shape=embed_dim)

        self.mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * embed_dim, embed_dim),
        )
        self.out_norm = torch.nn.LayerNorm(normalized_shape=embed_dim)
        # resize track dim back to 2
        self.resize_dim = torch.nn.Linear(embed_dim, 2)
        # resize n_track to n_waypoint
        self.resize_track = torch.nn.Linear(n_track, n_waypoints)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # concat the tracks on the last dim.
        # output size: (B, n_track, 4)
        track = torch.cat([track_left, track_right], dim=2)
        # output size: (B, n_track, embed_dim)
        track = self.projection(track)
        track = self.in_norm(track)

        # self attention
        track = track + self.mha(track, track, track)[0]
        track = track + self.mlp(self.out_norm(track))

        # output size: (B, n_track, n_waypoint)
        x = self.resize_dim(track)
        # (B, n_waypoint, n_track)
        x = x.permute(0, 2, 1)
        x = self.resize_track(x)
        x = x.permute(0, 2, 1)
        return x


class TransformerPlanner(nn.Module):
    class Block(nn.Module):
        def __init__(
            self,
            d_model: int = 64,
            num_heads=8,
        ):
            super().__init__()

            self.in_norm = nn.LayerNorm(d_model)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(d_model, d_model * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * d_model, d_model),
            )
            self.mha = nn.MultiheadAttention(d_model, num_heads=num_heads)
            self.out_norm = nn.LayerNorm(d_model)

        def forward(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:
            # x shape: (B, n_track, d_model)
            x = self.in_norm(x)
            x = x + self.mha(x, x, x)[0]
            x = x + self.mlp(self.out_norm(x))
            
            # output size: (B, n_track, d_model)
            return x

    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 32,
        num_heads = 8,
        n_blocks = 15,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_blocks = n_blocks

        self.query_embed = nn.Embedding(n_waypoints, d_model)
        self.track_pos_embed = nn.Parameter(torch.zeros(1, n_track, d_model))
        
        # to transform last dim of tracks to d_model
        # 2 * 2 because concating left and right track
        self.projection = torch.nn.Linear(2 * 2, d_model)

        self.mhas = nn.ModuleList()
        self.blocks = nn.ModuleList()
        
        for _ in range(n_blocks):
            self.mhas.append(
                # assuming all q k v have same dims
                nn.MultiheadAttention(
                    embed_dim=d_model,
                    kdim=d_model,
                    vdim=d_model,
                    num_heads=num_heads,
                    batch_first=True
                )
            )
            self.blocks.append(TransformerPlanner.Block(d_model, num_heads))
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, n_blocks)

        self.resizer = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # concat the tracks on the last dim.
        # output size: (B, n_track, 4)
        track = torch.cat([track_left, track_right], dim=2)
        # output size: (B, n_track, d_model)
        track = self.projection(track) + self.track_pos_embed
        
        indices = torch.arange(self.n_waypoints).to(device=track.device)
        # output shape: (n_waypoint, d_model)
        query = self.query_embed(indices)
        
        # Add the batch dim: (1, n_waypoint, d_model)
        query = query.unsqueeze(0)
        # Repeat on the batch dim: (B, n_waypoint, d_model)
        query = query.repeat(track.shape[0], 1, 1)

        # shape: (B, n_waypoints, d_models)
        query = self.transformer(tgt=query, memory=track)
        # shape: (B, n_waypoints, 2)
        x = self.resizer(query)
        return x

# if __name__ == "__main__":
#     tp = TransformerPlanner()
#     print(tp)
#     tp.forward(torch.rand(2, 10, 2), torch.rand(2, 10, 2))

class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        raise NotImplementedError


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)
    print(f"{model_size_mb=}")

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
