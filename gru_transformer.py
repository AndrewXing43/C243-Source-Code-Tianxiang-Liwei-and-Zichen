import torch
from torch import nn

from .augmentations import GaussianSmoothing

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_distance=500):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        # learned vector b ∈ ℝ^{2L−1}
        self.bias = nn.Parameter(torch.zeros(num_heads, 2 * max_distance - 1))

    def forward(self, seq_len):
        # positions i,j
        ctx = torch.arange(seq_len, device=self.bias.device)
        rel = ctx[None, :] - ctx[:, None]  # (T,T)

        rel = rel.clamp(-self.max_distance + 1, self.max_distance - 1)
        rel_index = rel + self.max_distance - 1  # shift to [0, 2L−2]

        # convert to bias matrix
        B = self.bias[:, rel_index]                 # (H, T,T)
        return B[None, :, :, :]               # (1,H,T,T) for broadcasting

class RelPosEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, max_distance=500, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = kwargs["nhead"]
        self.rel_pos_bias = RelativePositionBias(kwargs["nhead"], max_distance)

    def forward(self, src, src_mask=None, is_causal=True, src_key_padding_mask=None):
        # src: (T,B,D)
        T, B, D = src.shape

        # ---- build causal mask M ----
        causal_mask = torch.full((T, T), float("-inf"), device=src.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)  # j > i → -inf

        # ---- relative positional bias B ----
        rel_bias = self.rel_pos_bias(T)  # (1,1,T,T)

        # ---- add masks inside attention ----
        combined_mask = causal_mask[:, :] + rel_bias[0]      # (H, T,T)
        # expand to (B*H, T, T)
        combined_mask = combined_mask
        combined_mask = combined_mask.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, T, T)
        combined_mask = combined_mask.reshape(B*self.num_heads, T, T)

        # MultiheadAttention expects mask as additive mask
        src2 = self.self_attn(
            src,
            src,
            src,
            attn_mask=combined_mask,
            need_weights=False,
        )[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class GRUDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
    ):
        super(GRUDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # GRU layers
        self.gru_decoder = nn.GRU(
            neural_dim * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
        
        
        encoder_layer = RelPosEncoderLayer(
            d_model=self.hidden_dim,
            nhead=3,
            dim_feedforward=self.hidden_dim * 2,
            dropout=self.dropout,
            activation="gelu",
            batch_first=False,
            max_distance=500) 
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )
                
        
        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):     # neuralInput.shape (batchsize=128, seqlen=859, feature=256)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)     # (batchsize=128, seqlen=859, feature=256)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)  # (batchsize=128, seqlen=859, feature=256)

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )       # (128, 207, 8192)

        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())   # (batchsize, seqlen, hidden_dim)
        
        hid = hid.permute(1, 0, 2)
        hid = self.transformer_encoder(hid).permute(1, 0, 2)
        
        # get seq
        seq_out = self.fc_decoder_out(hid)
        return seq_out
