#!/usr/bin/env python3
"""Compare Pre-LN and Post-LN gradients in a tiny causal Transformer.

For a given depth, the script:
  1. Initializes the two architectures with identical shared parameters.
  2. Runs one forward pass and one backward pass on the same token batch.
  3. Saves layer-wise gradient norms, weight/gradient histograms, and a matched
     element-wise gradient scatter plot.
  4. Exports JSON data used by the companion study page.

Example:
    python experiments/pre_post_ln_demo.py --n-layers 24 --output-dir results/depth24
    python experiments/pre_post_ln_demo.py --n-layers 32 --output-dir results/depth32
    python experiments/pre_post_ln_demo.py --n-layers 48 --output-dir results/depth48
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, eps: float = 1e-5):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=0.0, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model, eps=eps)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor, pre_ln: bool):
        if pre_ln:
            h = self.ln1(x)
            # h is passed as Q/K/V input for self-attention. PyTorch applies
            # distinct learned Q, K, and V projections internally.
            attn_out, _ = self.attn(
                query=h, key=h, value=h, attn_mask=causal_mask, need_weights=False
            )
            x = x + attn_out

            h = self.ln2(x)
            ff_out = self.ff2(self.act(self.ff1(h)))
            x = x + ff_out
        else:
            # The same input enters self-attention, but Q/K/V use separate
            # learned projection matrices inside nn.MultiheadAttention.
            attn_out, _ = self.attn(
                query=x, key=x, value=x, attn_mask=causal_mask, need_weights=False
            )
            x = self.ln1(x + attn_out)

            ff_out = self.ff2(self.act(self.ff1(x)))
            x = self.ln2(x + ff_out)
        return x


class TinyCausalLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 257,
        max_seq_len: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 8,
        pre_ln: bool,
    ):
        super().__init__()
        self.pre_ln = pre_ln
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        # Pre-LN conventionally uses a final normalization before the LM head.
        # Post-LN's last block already ends with LayerNorm.
        self.final_norm = nn.LayerNorm(d_model) if pre_ln else nn.Identity()
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)[None, :, :]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        for block in self.blocks:
            x = block(x, causal_mask, pre_ln=self.pre_ln)
        if self.pre_ln:
            x = self.final_norm(x)
        return self.lm_head(x)


def collect_gradients(model: TinyCausalLM) -> dict:
    ff2_by_layer = [b.ff2.weight.grad.detach().cpu().clone() for b in model.blocks]
    qkv_by_layer = [
        b.attn.in_proj_weight.grad.detach().cpu().clone() for b in model.blocks
    ]
    return {
        "ff2_weight_by_layer": ff2_by_layer,
        "attn_qkv_weight_by_layer": qkv_by_layer,
        "final_ff2_weight_grad": ff2_by_layer[-1].numpy().ravel(),
        "final_ff2_weight": model.blocks[-1].ff2.weight.detach().cpu().numpy().ravel(),
    }


def norm_list(tensors: list[torch.Tensor]) -> list[float]:
    return [float(t.norm().item()) for t in tensors]


def histogram_density(values: np.ndarray, bins: np.ndarray) -> list[float]:
    return np.histogram(values, bins=bins, density=True)[0].astype(float).tolist()


def save_experiment(
    gradients: dict[str, dict],
    n_layers: int,
    output_dir: Path,
    web_data_path: Path | None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_ids = np.arange(1, n_layers + 1)
    post = gradients["Post-LN"]
    pre = gradients["Pre-LN"]

    # Plot 1: layer-wise gradient norms across the whole decoder stack.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, key, title in [
        (axes[0], "ff2_weight_by_layer", "FFN second linear weight (W₂)"),
        (axes[1], "attn_qkv_weight_by_layer", "Attention QKV projection weights"),
    ]:
        ax.plot(layer_ids, norm_list(post[key]), marker="o", label="Post-LN")
        ax.plot(layer_ids, norm_list(pre[key]), marker="o", label="Pre-LN")
        ax.set_title(title)
        ax.set_xlabel("Decoder block (1 = first / bottom block)")
        ax.set_ylabel("Gradient L2 norm")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    fig.suptitle("Layer-wise gradient norms after one backward pass")
    fig.savefig(output_dir / "layerwise_gradient_norms.png", dpi=180)
    fig.savefig(output_dir / "layerwise_gradient_norms.svg")
    plt.close(fig)

    post_g = post["final_ff2_weight_grad"]
    pre_g = pre["final_ff2_weight_grad"]
    weights = post["final_ff2_weight"]
    assert weights.shape == post_g.shape == pre_g.shape

    corr = float(np.corrcoef(post_g, pre_g)[0, 1])
    rng = np.random.default_rng(23)
    plot_count = min(12000, post_g.size)
    point_indices = rng.choice(post_g.size, size=plot_count, replace=False)
    point_limit = float(np.quantile(np.abs(np.concatenate([post_g, pre_g])), 0.995))
    point_limit = max(point_limit, 1e-8)

    # Plot 2: weights and gradients on shared signed and log-magnitude scales.
    eps = 1e-30
    weights_log = np.log10(np.abs(weights) + eps)
    post_log = np.log10(np.abs(post_g) + eps)
    pre_log = np.log10(np.abs(pre_g) + eps)

    signed_limit = float(np.max(np.abs(np.concatenate([weights, post_g, pre_g])))) * 1.02
    signed_bins = np.linspace(-signed_limit, signed_limit, 101)
    log_pool = np.concatenate([weights_log, post_log, pre_log])
    log_bins = np.linspace(float(log_pool.min()), float(log_pool.max()), 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    ax = axes[0]
    ax.hist(weights, bins=signed_bins, density=True, histtype="step", linewidth=1.8,
            label="Shared weights (Post-/Pre-LN)")
    ax.hist(post_g, bins=signed_bins, density=True, histtype="step", linewidth=1.8,
            label="Post-LN gradients")
    ax.hist(pre_g, bins=signed_bins, density=True, histtype="step", linewidth=1.8,
            linestyle="--", label="Pre-LN gradients")
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlim(-signed_limit, signed_limit)
    ax.set_xlabel("Signed element value (common linear scale)")
    ax.set_ylabel("Density")
    ax.set_title("Signed values: weights vs gradients")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(weights_log, bins=log_bins, density=True, histtype="step", linewidth=2,
            label=f"Shared weights (median {np.median(np.abs(weights)):.2e})")
    ax.hist(post_log, bins=log_bins, density=True, histtype="step", linewidth=2,
            label=f"Post-LN gradients (median {np.median(np.abs(post_g)):.2e})")
    ax.hist(pre_log, bins=log_bins, density=True, histtype="step", linewidth=2,
            linestyle="--", label=f"Pre-LN gradients (median {np.median(np.abs(pre_g)):.2e})")
    ticks = np.arange(np.floor(log_pool.min()), np.ceil(log_pool.max()) + 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"1e{int(t)}" for t in ticks])
    ax.set_xlabel("Absolute element magnitude (log scale)")
    ax.set_ylabel("Density per log-magnitude interval")
    ax.set_title("Magnitude distributions: weights vs gradients")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.suptitle("Final FFN W₂: parameter values vs their gradients")
    fig.savefig(output_dir / "weight_gradient_histogram.png", dpi=180)
    fig.savefig(output_dir / "weight_gradient_histogram.svg")
    plt.close(fig)

    # Plot 3: matched elements; x/y are gradients of the same matrix elements.
    fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    ax.scatter(post_g[point_indices], pre_g[point_indices], s=8, alpha=0.22, rasterized=True)
    ax.plot([-point_limit, point_limit], [-point_limit, point_limit], "--", linewidth=1.5, label="y = x")
    ax.set_xlim(-point_limit, point_limit)
    ax.set_ylim(-point_limit, point_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Post-LN gradient for final FFN W₂ element")
    ax.set_ylabel("Pre-LN gradient for same W₂ element")
    ax.set_title(
        f"Matched element-wise gradients (full-matrix Pearson r = {corr:.3f})\\n"
        f"Showing {plot_count:,} of {post_g.size:,} elements; outer 0.5% clipped"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "gradient_scatter.png", dpi=180)
    fig.savefig(output_dir / "gradient_scatter.svg")
    plt.close(fig)

    # Compact per-depth data consumed by the study page.
    compact_signed_bins = np.linspace(-signed_limit, signed_limit, 31)
    compact_log_bins = np.linspace(float(log_pool.min()), float(log_pool.max()), 31)
    scatter_count = min(80, post_g.size)
    scatter_indices = np.random.default_rng(23).choice(post_g.size, size=scatter_count, replace=False)
    web_data = {
        "n_layers": int(n_layers),
        "signed": {
            "edges": compact_signed_bins.astype(float).tolist(),
            "series": {
                "weights": histogram_density(weights, compact_signed_bins),
                "post_ln_gradients": histogram_density(post_g, compact_signed_bins),
                "pre_ln_gradients": histogram_density(pre_g, compact_signed_bins),
            },
        },
        "log_abs": {
            "edges": compact_log_bins.astype(float).tolist(),
            "series": {
                "weights": histogram_density(weights_log, compact_log_bins),
                "post_ln_gradients": histogram_density(post_log, compact_log_bins),
                "pre_ln_gradients": histogram_density(pre_log, compact_log_bins),
            },
        },
        "layerwise": {
            "post_ln": {
                "ff2": norm_list(post["ff2_weight_by_layer"]),
                "attn_qkv": norm_list(post["attn_qkv_weight_by_layer"]),
            },
            "pre_ln": {
                "ff2": norm_list(pre["ff2_weight_by_layer"]),
                "attn_qkv": norm_list(pre["attn_qkv_weight_by_layer"]),
            },
        },
        "scatter": {
            "post_ln": np.round(post_g[scatter_indices], 7).tolist(),
            "pre_ln": np.round(pre_g[scatter_indices], 7).tolist(),
            "sample_size": int(scatter_count),
            "total_size": int(post_g.size),
            "pearson_r": corr,
            "axis_limit": point_limit,
        },
        "summary": {
            "weight_median_abs": float(np.median(np.abs(weights))),
            "post_ln_grad_median_abs": float(np.median(np.abs(post_g))),
            "pre_ln_grad_median_abs": float(np.median(np.abs(pre_g))),
        },
    }
    json_text = json.dumps(web_data, separators=(",", ":"))
    (output_dir / "pre_post_ln_web_histogram_data.json").write_text(json_text, encoding="utf-8")
    if web_data_path is not None:
        web_data_path.parent.mkdir(parents=True, exist_ok=True)
        web_data_path.write_text(json_text, encoding="utf-8")

    print(f"Depth: {n_layers}")
    print(f"Final FFN W₂ gradient norm — Post-LN: {torch.tensor(post['ff2_weight_by_layer'][-1]).norm().item():.6e}")
    print(f"Final FFN W₂ gradient norm — Pre-LN : {torch.tensor(pre['ff2_weight_by_layer'][-1]).norm().item():.6e}")
    print(f"Full-matrix gradient Pearson r: {corr:.4f}")
    print(f"Plots and JSON written to: {output_dir}")
    if web_data_path is not None:
        print(f"Website JSON written to: {web_data_path}")
    return web_data


def run_experiment(n_layers: int, device: str = "cpu") -> dict[str, dict]:
    torch.manual_seed(7)
    vocab_size, batch_size, seq_len = 257, 8, 32

    pre = TinyCausalLM(
        vocab_size=vocab_size, max_seq_len=seq_len, d_model=128,
        n_heads=4, d_ff=512, n_layers=n_layers, pre_ln=True
    ).to(device)
    post = TinyCausalLM(
        vocab_size=vocab_size, max_seq_len=seq_len, d_model=128,
        n_heads=4, d_ff=512, n_layers=n_layers, pre_ln=False
    ).to(device)

    # Copy all shared parameter values from Pre-LN to Post-LN. The only extra
    # learned parameters are Pre-LN's final LayerNorm weight and bias.
    state = pre.state_dict()
    load_result = post.load_state_dict(state, strict=False)
    assert all("final_norm" in key for key in load_result.unexpected_keys)
    assert not load_result.missing_keys

    pre_params, post_params = dict(pre.named_parameters()), dict(post.named_parameters())
    common = sorted(set(pre_params) & set(post_params))
    assert all(torch.equal(pre_params[key], post_params[key]) for key in common)

    torch.manual_seed(101)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)
    input_ids, targets = tokens[:, :-1], tokens[:, 1:]

    results = {}
    for name, model in [("Post-LN", post), ("Pre-LN", pre)]:
        model.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()

        ff2_by_layer = [b.ff2.weight.grad.detach().cpu().clone() for b in model.blocks]
        qkv_by_layer = [b.attn.in_proj_weight.grad.detach().cpu().clone() for b in model.blocks]
        results[name] = {
            "loss": float(loss.item()),
            "ff2_weight_by_layer": ff2_by_layer,
            "attn_qkv_weight_by_layer": qkv_by_layer,
            "final_ff2_weight_grad": ff2_by_layer[-1].numpy().ravel(),
            "final_ff2_weight": model.blocks[-1].ff2.weight.detach().cpu().numpy().ravel(),
        }
    print(f"Shared parameter tensors equal at initialization: {len(common)}")
    print(f"Losses: Post-LN={results['Post-LN']['loss']:.6f}; Pre-LN={results['Pre-LN']['loss']:.6f}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-layers", type=int, default=24, help="Decoder depth.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/depth24"))
    parser.add_argument("--web-data-path", type=Path, default=None,
                        help="Optional path to write the compact JSON consumed by the webpage.")
    parser.add_argument("--device", choices=("cpu",), default="cpu",
                        help="CPU is recommended for deterministic reproduction.")
    args = parser.parse_args()
    if args.n_layers < 1:
        parser.error("--n-layers must be at least 1")

    gradients = run_experiment(args.n_layers, args.device)
    save_experiment(gradients, args.n_layers, args.output_dir, args.web_data_path)


if __name__ == "__main__":
    main()
