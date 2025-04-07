import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.lib import HyperParams

LOG_TERMINAL = False  

class Encoder(nn.Module):
    def __init__(self, p):
        super().__init__()
        layers = [EncoderLayer(p) for _ in range(p.n_L)]
        self.layers = nn.ModuleList(layers)

    def forward(self, src, src_mask):
        # src = [batch_size, src_seq_len, d_x]
        # src_mask = [batch_size, 1, 1, src_seq_len]
        debug("\nencoder")
        debug("src: ", src.shape, is_nan_string(src))
        debug("src_mask: ", src_mask.shape, is_nan_string(src_mask))

        for layer in self.layers:
            src = layer(src, src_mask)
        return src
  

class EncoderLayer(nn.Module):
    def __init__(self, p):
        super().__init__()
        d_h = p.d_x

        # Sublayer 1: Performer Attention
        self.layernorm1 = nn.LayerNorm(d_h)
        self.performer_attn = PerformerAttention(p)  # Replace SelfAttention
        self.dropout1 = nn.Dropout(p.dropout)

        # Sublayer 2: Feed-Forward
        self.layernorm2 = nn.LayerNorm(d_h)
        self.ffn = nn.Sequential(
            nn.Linear(d_h, p.d_f),
            nn.ReLU(),
            nn.Linear(p.d_f, d_h)
        )
        self.dropout2 = nn.Dropout(p.dropout)

    def forward(self, src, src_mask):
        # src = [batch_size, src_seq_len, d_x]
        # src_mask = [batch_size, 1, 1, src_seq_len]

        # Sublayer 1: Performer Attention
        z = self.layernorm1(src)
        z = self.performer_attn(z, z, z, src_mask)
        z = self.dropout1(z)
        src = src + z

        # Sublayer 2: Feed-Forward
        z = self.layernorm2(src)
        z = self.ffn(z)
        z = self.dropout2(z)
        src = src + z

        return src
  
class EmbeddingSinusoidal(nn.Module):
    def __init__(self, d_vocab, d_x, dropout, max_length):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_x = d_x
        self.max_length = max_length

        # Token embedding
        self.tok_embedding = nn.Embedding(d_vocab, d_x)
        self.scale = torch.sqrt(torch.tensor([d_x], dtype=torch.float))

        # Sinusoidal positional encoding
        pe = torch.zeros(max_length, d_x)
        position = torch.arange(0., max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_x, 2) * (-math.log(10000.0) / d_x))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        self.reset_parameters()

    def forward(self, src):
        # src = [batch_size, src_seq_len]
        tok_emb = self.tok_embedding(src) * self.scale.to(src.device)
        pos_emb = self.pe[:, :src.size(1)].to(src.device)
        x = tok_emb + pos_emb
        return self.dropout(x)  # [batch_size, src_seq_len, d_x]

    def reset_parameters(self):
        nn.init.normal_(self.tok_embedding.weight, mean=0, std=1./math.sqrt(self.d_x))


def is_nan_string(x):
    if not LOG_TERMINAL:
        return "skip"
    return "NaN" if torch.isnan(x).any() else "ok"

def debug(*args):
    if LOG_TERMINAL:
        print(*args)

def build_performer(params, pad_idx):
    p = HyperParams()
    p.d_vocab = params.input_dim  # Vocabulary size (e.g., 512)
    p.d_pos = 509  # Max sequence length (from Task 1.2 data)

    p.d_f = params.filter  # Feed-forward filter size
    p.n_L = params.n_layers  # Number of layers (e.g., 4)
    p.n_I = params.n_heads  # Number of attention heads (e.g., 8)

    p.d_x = params.hidden  # Hidden dimension (e.g., 512)
    p.d_k = p.d_x // p.n_I  # Key/query dimension per head
    p.d_v = p.d_x // p.n_I  # Value dimension per head

    p.dropout = params.dropout  # Dropout rate (e.g., 0.1)

    embedding = EmbeddingSinusoidal(d_vocab=params.input_dim,
                                    d_x=p.d_x,
                                    dropout=params.dropout,
                                    max_length=509)
    encoder = Encoder(p=p)
    decoder = Decoder(p=p)
    model = Seq2Seq(p=p,
                    embedding=embedding,
                    encoder=encoder,
                    decoder=decoder,
                    pad_idx=pad_idx)

    return model

class PerformerAttention(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.d_h = p.d_x
        self.n_I = p.n_I
        self.d_k = p.d_k
        self.d_v = p.d_v

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(self.d_h, p.d_k * p.n_I)
        self.W_k = nn.Linear(self.d_h, p.d_k * p.n_I)
        self.W_v = nn.Linear(self.d_h, p.d_v * p.n_I)
        self.W_o = nn.Linear(p.d_v * p.n_I, p.d_x)

        self.dropout = nn.Dropout(p.dropout)
        self.scale = math.sqrt(p.d_k)

        # Random features for Favor+ (simplified)
        self.num_features = 256  # Number of random features (configurable)
        self.random_features = nn.Parameter(torch.randn(p.d_k, self.num_features))

    def forward(self, query, key, value, mask=None):
        # query, key, value = [batch_size, seq_len, d_h]
        # mask = [batch_size, 1, 1, seq_len] (for encoder) or [batch_size, 1, seq_len, seq_len] (decoder)

        bsz = query.shape[0]
        seq_len = query.shape[1]

        # Linear projections
        Q = self.W_q(query)  # [bsz, seq_len, n_I * d_k]
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(bsz, seq_len, self.n_I, self.d_k).permute(0, 2, 1, 3)  # [bsz, n_I, seq_len, d_k]
        K = K.view(bsz, seq_len, self.n_I, self.d_k).permute(0, 2, 1, 3)
        V = V.view(bsz, seq_len, self.n_I, self.d_v).permute(0, 2, 1, 3)

        # Favor+ approximation (simplified kernel-based attention)
        Q_rf = F.softmax(Q @ self.random_features / self.scale, dim=-1)  # [bsz, n_I, seq_len, num_features]
        K_rf = F.softmax(K @ self.random_features / self.scale, dim=-1)

        # Compute prefix sums for efficiency
        KV = torch.einsum('bhjd,bhjf->bhdf', V, K_rf)  # [bsz, n_I, d_v, num_features]
        attention = torch.einsum('bhif,bhdf->bhid', Q_rf, KV)  # [bsz, n_I, seq_len, d_v]

        # Reshape and project
        attention = attention.permute(0, 2, 1, 3).contiguous()  # [bsz, seq_len, n_I, d_v]
        attention = attention.view(bsz, seq_len, self.n_I * self.d_v)
        out = self.W_o(attention)  # [bsz, seq_len, d_x]

        if mask is not None:
            out = out.masked_fill(mask.squeeze(1).squeeze(1) == 0, 0)

        return self.dropout(out)


class Decoder(nn.Module):
    def __init__(self, p):
        super().__init__()
        layers = [DecoderLayer(p) for _ in range(p.n_L)]
        self.layers = nn.ModuleList(layers)

    def forward(self, trg, src, trg_mask, src_mask):
        # trg = [batch_size, trg_seq_len, d_x]
        # src = [batch_size, src_seq_len, d_x]
        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)
        return trg


class DecoderLayer(nn.Module):
    def __init__(self, p):
        super().__init__()
        d_h = p.d_x

        # Sublayer 1: Self-Attention
        self.layernorm1 = nn.LayerNorm(d_h)
        self.self_attn = PerformerAttention(p)
        self.dropout1 = nn.Dropout(p.dropout)

        # Sublayer 2: Encoder-Decoder Attention
        self.layernorm2 = nn.LayerNorm(d_h)
        self.enc_attn = PerformerAttention(p)
        self.dropout2 = nn.Dropout(p.dropout)

        # Sublayer 3: Feed-Forward
        self.layernorm3 = nn.LayerNorm(d_h)
        self.ffn = nn.Sequential(
            nn.Linear(d_h, p.d_f),
            nn.ReLU(),
            nn.Linear(p.d_f, d_h)
        )
        self.dropout3 = nn.Dropout(p.dropout)

    def forward(self, trg, src, trg_mask, src_mask):
        # Self-Attention
        z = self.layernorm1(trg)
        z = self.self_attn(z, z, z, trg_mask)
        z = self.dropout1(z)
        trg = trg + z

        # Encoder-Decoder Attention
        z = self.layernorm2(trg)
        z = self.enc_attn(z, src, src, src_mask)
        z = self.dropout2(z)
        trg = trg + z

        # Feed-Forward
        z = self.layernorm3(trg)
        z = self.ffn(z)
        z = self.dropout3(z)
        trg = trg + z

        return trg


class Seq2Seq(nn.Module):
    def __init__(self, p, embedding, encoder, decoder, pad_idx):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.p = p
        self.output_layer = nn.Linear(p.d_x, p.d_vocab)  # Replace KAN with linear
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def make_masks(self, src, trg):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)  # [bsz, 1, 1, src_len]
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask  # [bsz, 1, trg_len, trg_len]
        return src_mask, trg_mask

    def forward(self, src, trg):
        src_mask, trg_mask = self.make_masks(src, trg)
        src_emb = self.embedding(src)
        trg_emb = self.embedding(trg)
        enc_src = self.encoder(src_emb, src_mask)
        dec_out = self.decoder(trg_emb, enc_src, trg_mask, src_mask)
        logits = self.output_layer(dec_out)  # [bsz, trg_seq_len, d_vocab]
        return logits

    def greedy_inference(self, src, sos_idx, eos_idx, max_length):
        self.eval()
        src = src.to(self.device)
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        src_emb = self.embedding(src)
        enc_src = self.encoder(src_emb, src_mask)
        trg = torch.ones(src.shape[0], 1).fill_(sos_idx).type_as(src).to(self.device)
        done = torch.zeros(src.shape[0], dtype=torch.bool).to(self.device)

        for _ in range(max_length):
            trg_emb = self.embedding(trg)
            trg_mask = self.make_masks(src, trg)[1]
            output = self.decoder(trg_emb, enc_src, trg_mask, src_mask)
            logits = self.output_layer(output)
            pred = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
            trg = torch.cat([trg, pred], dim=1)
            done |= (pred.squeeze(1) == eos_idx)
            if done.all():
                break
        return trg

    def validate_with_sympy(self, output_tokens, ground_truth_tokens, token_to_str_map):
        """Post-process output with SymPy for empirical consistency (pseudo-code)."""
        import sympy as sp
        # Convert token sequences to strings (requires external token mapping)
        pred_str = " ".join([token_to_str_map[t.item()] for t in output_tokens[0]])
        true_str = " ".join([token_to_str_map[t.item()] for t in ground_truth_tokens[0]])
        
        # Parse with SymPy (assumes HEP symbols defined externally)
        pred_expr = sp.sympify(pred_str, evaluate=False)
        true_expr = sp.sympify(true_str, evaluate=False)
        
        # Check equivalence (simplified example)
        is_equiv = sp.simplify(pred_expr - true_expr) == 0
        return is_equiv, pred_expr
