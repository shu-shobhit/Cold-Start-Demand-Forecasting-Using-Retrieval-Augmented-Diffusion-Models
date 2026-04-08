"""Diffusion backbone modules used by the RATD forecasting model.

The code in this file implements the denoising network, residual blocks,
diffusion-step embeddings, and the retrieval-aware attention mechanism that
injects retrieved reference futures into the forecasting process.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import Attention as CrossAttention, FeedForward, AdaLayerNorm
from einops import rearrange, repeat
from linear_attention_transformer import LinearAttentionTransformer
from torch import einsum

def default(val, d):
    """Return ``val`` when present, otherwise evaluate the fallback value.

    Args:
        val: Candidate value that may be ``None``.
        d: Fallback value used when ``val`` is ``None``.

    Returns:
        Any: ``val`` if it is not ``None``, otherwise ``d``.
    """

    return val if val is not None else d

class ReferenceModulatedCrossAttention(nn.Module):
    """Cross-attention module that fuses latent states with retrieved futures.

    Args:
        dim: Feature dimension of the latent state.
        heads: Number of attention heads.
        dim_head: Per-head hidden size.
        context_dim: Dimension of the conditioning context.
        dropout: Dropout probability applied to attention weights.
        talking_heads: Whether to use learned head mixing.
        prenorm: Whether to normalize inputs before projection.

    Returns:
        None: The constructor initializes the attention module in-place.
    """

    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        context_dim = None,
        dropout = 0.,
        talking_heads = False,
        prenorm = False
    ):
        """Initialize the reference-modulated attention layer.

        Args:
            dim: Feature dimension of the latent state.
            heads: Number of attention heads.
            dim_head: Per-head hidden size.
            context_dim: Dimension of the conditioning context.
            dropout: Dropout probability applied to attention weights.
            talking_heads: Whether to use learned head mixing.
            prenorm: Whether to normalize inputs before projection.

        Returns:
            None: The constructor initializes the attention module in-place.
        """

        super().__init__()
        context_dim = default(context_dim, dim)

        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)

        # The current implementation projects the denoising state into queries
        # and constructs keys / values from a mix of current state, side
        # information, and retrieved references.
        self.y_to_q = nn.Linear(dim, inner_dim, bias = False)
        self.cond_to_k = nn.Linear(2*dim+context_dim, inner_dim, bias = False)
        self.ref_to_v = nn.Linear(dim+context_dim, inner_dim, bias = False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()

    def forward(
        self,
        x,
        cond_info,
        reference,
        return_attn = False,
    ):
        """Apply reference-aware attention to the latent state.

        Args:
            x: Latent state tensor with shape ``(B, C, K, L)``.
            cond_info: Conditioning tensor aligned with ``x``.
            reference: Retrieved reference tensor with shape ``(B, K, R)``.
            return_attn: Whether to also return raw attention weights.

        Returns:
            torch.Tensor | tuple: Fused output tensor, optionally with
            auxiliary attention outputs when ``return_attn`` is ``True``.
        """

        B, C, K, L, h, device = x.shape[0], x.shape[1], x.shape[2], x.shape[-1], self.heads, x.device
        x = self.norm(x)
        reference = self.norm(reference)
        cond_info = self.context_norm(cond_info)

        # Repeat the retrieved references across channels so each denoising
        # channel can attend to the same set of candidate future patterns.
        reference = repeat(reference, 'b n c -> (b f) n c', f=C)# (B*C, K, L)
        q_y = self.y_to_q(x.reshape(B*C, K, L))# (B*C,K,ND)

        cond=self.cond_to_k(torch.cat((x.reshape(B*C, K, L), cond_info.reshape(B*C, K, L), reference), dim=-1))# (B*C,K,ND)
        ref=self.ref_to_v(torch.cat((x.reshape(B*C, K, L), reference), dim=-1))# (B*C,K,ND)
        q_y, cond, ref = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q_y, cond, ref))# (B*C, N, K, D)

        # Similarity is computed between the reference-conditioned keys and
        # values, then normalized in both directions for asymmetric outputs.
        sim = einsum('b h i d, b h j d -> b h i j', cond, ref) * self.scale # (B*C, N, K, K)
        attn = sim.softmax(dim = -1)
        context_attn = sim.softmax(dim = -2)
        # dropouts
        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)
        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, ref)
        context_out = einsum('b h j i, b h j d -> b h i d', context_attn, cond)
        # merge heads and combine out
        out, context_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, context_out))
        out = self.to_out(out)
        if return_attn:
            return out, context_out, attn, context_attn

        return out


def get_torch_trans(heads=8, layers=1, channels=64):
    """Build a standard transformer encoder for one mixing axis.

    Args:
        heads: Number of attention heads.
        layers: Number of encoder layers.
        channels: Hidden size of the transformer.

    Returns:
        nn.TransformerEncoder: Transformer encoder used in residual blocks.
    """

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):
    """Build a linear-attention transformer for one mixing axis.

    Args:
        heads: Number of attention heads.
        layers: Number of transformer layers.
        channels: Hidden size of the transformer.
        localheads: Unused legacy argument retained for compatibility.
        localwindow: Unused legacy argument retained for compatibility.

    Returns:
        LinearAttentionTransformer: Linear-attention mixing module.
    """

    return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0, 
        local_attn_window_size = 0,
    )

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    """Create a 1D convolution with Kaiming-initialized weights.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel width.

    Returns:
        nn.Conv1d: Initialized convolution layer.
    """

    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def Reference_Modulated_Attention(in_channels, out_channels, kernel_size):
    """Create a reference-fusion convolution helper.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel width.

    Returns:
        nn.Conv1d: Initialized convolution layer.
    """

    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class DiffusionEmbedding(nn.Module):
    """Sinusoidal embedding lookup for discrete diffusion timesteps.

    Args:
        num_steps: Number of diffusion steps in the schedule.
        embedding_dim: Size of the base sinusoidal embedding.
        projection_dim: Optional hidden size after projection.

    Returns:
        None: The constructor initializes projection layers and buffers.
    """

    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        """Initialize the timestep embedding lookup and projections.

        Args:
            num_steps: Number of diffusion steps in the schedule.
            embedding_dim: Size of the base sinusoidal embedding.
            projection_dim: Optional hidden size after projection.

        Returns:
            None: The constructor initializes projection layers and buffers.
        """

        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        """Project discrete diffusion steps into learned embeddings.

        Args:
            diffusion_step: Tensor of diffusion step indices.

        Returns:
            torch.Tensor: Projected timestep embeddings.
        """

        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        """Construct the fixed sinusoidal embedding table.

        Args:
            num_steps: Number of diffusion steps.
            dim: Half of the final sinusoidal embedding width.

        Returns:
            torch.Tensor: Lookup table with shape ``(num_steps, 2 * dim)``.
        """

        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_RATD(nn.Module):
    """Diffusion denoiser used to predict noise over the forecast window.

    Args:
        config: Diffusion configuration dictionary.
        inputdim: Number of input channels supplied by the wrapper model.
        use_ref: Whether retrieval references are expected by the backbone.

    Returns:
        None: The constructor initializes the denoising network in-place.
    """

    def __init__(self, config, inputdim=2, use_ref=True):
        """Initialize the diffusion denoiser backbone.

        Args:
            config: Diffusion configuration dictionary.
            inputdim: Number of input channels supplied by the wrapper model.
            use_ref: Whether retrieval references are expected by the backbone.

        Returns:
            None: The constructor initializes the denoising network in-place.
        """

        super().__init__()
        self.channels = config["channels"]
        self.use_ref=use_ref
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        # Each residual block receives the same side information and optional
        # references, then contributes one skip tensor to the final prediction.
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    ref_size=config["ref_size"],
                    h_size=config["h_size"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step, reference=None):
        """Run the denoiser for one diffusion step.

        Args:
            x: Input tensor with shape ``(B, inputdim, K, L)``.
            cond_info: Side-information tensor with shape ``(B, C, K, L)``.
            diffusion_step: Current diffusion step indices.
            reference: Optional retrieval tensor.

        Returns:
            torch.Tensor: Predicted noise tensor with shape ``(B, K, L)``.
        """

        B, inputdim, K, L = x.shape

        # Flatten the spatial layout temporarily so 1x1 convolutions can act as
        # per-position channel projections.
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb, reference)
            skip.append(skip_connection)

        # Aggregate all skip paths before the final projection back to one
        # predicted noise channel per variable and timestep.
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    """Residual denoising block with time/feature mixing and retrieval fusion.

    Args:
        side_dim: Channel count of the side-information tensor.
        ref_size: Forecast horizon length used by the reference branch.
        h_size: Observed history length used by the latent state.
        channels: Hidden channel size for the block.
        diffusion_embedding_dim: Size of the timestep embedding.
        nheads: Number of attention heads in the transformers.
        is_linear: Whether to use linear attention instead of PyTorch attention.

    Returns:
        None: The constructor initializes the residual block in-place.
    """

    def __init__(self, side_dim, ref_size, h_size, channels, diffusion_embedding_dim, nheads, is_linear=False):
        """Initialize one residual denoising block.

        Args:
            side_dim: Channel count of the side-information tensor.
            ref_size: Forecast horizon length used by the reference branch.
            h_size: Observed history length used by the latent state.
            channels: Hidden channel size for the block.
            diffusion_embedding_dim: Size of the timestep embedding.
            nheads: Number of attention heads in the transformers.
            is_linear: Whether to use linear attention instead of PyTorch attention.

        Returns:
            None: The constructor initializes the residual block in-place.
        """

        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        dim_heads=8
        self.fusion_type=1
        self.q_dim=nheads*dim_heads
        self.attn1 = CrossAttention(
                query_dim=nheads*dim_heads,
                heads=nheads,
                dim_head=dim_heads,
                dropout=0,
                bias=False,
            )
        self.RMA=ReferenceModulatedCrossAttention(dim=ref_size+h_size,context_dim=ref_size*3)
        self.line= nn.Linear(
                ref_size*3, ref_size+h_size
            )
        #self.line3 = nn.Linear(nheads*dim_heads, 2)

        self.is_linear = is_linear
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
    def forward_time(self, y, base_shape):
        """Mix information along the temporal axis for each feature.

        Args:
            y: Flattened hidden tensor with shape ``(B, C, K * L)``.
            base_shape: Original tensor shape tuple ``(B, C, K, L)``.

        Returns:
            torch.Tensor: Tensor after temporal mixing.
        """

        B, channel, K, L = base_shape
        if L == 1:
            return y

        # Rearrange to ``(B * K, L, C)`` or equivalent so each feature series is
        # processed independently across time.
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y
    def forward_feature(self, y, base_shape):
        """Mix information across features for each timestep.

        Args:
            y: Flattened hidden tensor with shape ``(B, C, K * L)``.
            base_shape: Original tensor shape tuple ``(B, C, K, L)``.

        Returns:
            torch.Tensor: Tensor after feature mixing.
        """

        B, channel, K, L = base_shape
        if K == 1:
            return y

        # Rearrange to ``(B * L, K, C)`` so the transformer can connect
        # variables observed at the same timestep.
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y
    
    def forward(self, x, cond_info, diffusion_emb, reference):
        """Apply one residual denoising update.

        Args:
            x: Hidden tensor with shape ``(B, C, K, L)``.
            cond_info: Side-information tensor with shape ``(B, S, K, L)``.
            diffusion_emb: Timestep embedding for the current diffusion step.
            reference: Optional retrieval tensor with retrieved futures.

        Returns:
            tuple: Updated residual state and skip tensor.
        """

        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        # Inject the current diffusion step so the block knows how much noise is
        # expected to remain in the sample.
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb
        #reference = repeat(reference, 'b n c -> (b f) n c', f=inputdim)
        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)

        if reference!=None and self.fusion_type==1:
            # Fusion type 1 replaces the projected side information with the
            # output of the custom reference-aware attention module.
            cond_info = self.RMA(y.reshape(B, channel, K, L),cond_info.reshape(B, channel, K, L),reference)
            #reference = self.line(reference)
            #reference = torch.sigmoid(reference)# (B,K,L)
            #reference=reference.reshape(B, 1, K, L).permute(0,1,3,2)
            #reference = repeat(reference, 'b a n c -> (b a f) n c', f=2*channel)# (B*2*channel, L,K)
            #cond_info = torch.bmm(cond_info.reshape(B*2*channel, K , L), reference)# (B*2*channel, K, K)
            #cond_info = torch.sigmoid(cond_info)
            #cond_info = torch.bmm(cond_info, y.reshape(B*2*channel,K, L)).reshape(B,2*channel,K*L)
            #y = y + cond_info
        elif reference!=None and self.fusion_type==2:
            # Fusion type 2 is a simpler additive reference path retained from
            # earlier experiments in the research code.
            reference = self.line(reference)
            reference = torch.sigmoid(reference)# (B,K,L)
            reference = reference.reshape(B, 1, K, L)
            reference = repeat(reference, 'b a n c -> b (a f) n c', f=channel)# (B*2*channel, L,K)
            cond_info = cond_info + reference.reshape(B, channel, K*L)
            
        y = y + cond_info.reshape(B, channel, K*L)

        # Alternate mixing across time and features before the gated residual
        # projection splits into residual and skip outputs.
        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        #y = y + cond_info.reshape(B, 2*channel, K*L)
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        # Split the output into a residual update for the next block and a skip
        # connection for the final denoising head.
        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
