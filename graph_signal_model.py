#!/usr/bin/env python3
"""
Graph Signal Model + Real Grok Embeddings (safe integration)
Run with:  XAI_API_KEY=your_key_here python3 graph_signal_model_grok.py
"""

from __future__ import annotations
import os
import time
import math
import random
import hashlib
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim

# -------- Grok API embedding client --------
import requests

def get_grok_embeddings(texts: list[str], api_key: str) -> np.ndarray:
    url = "https://api.x.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "grok-beta",
        "input": texts
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    embeddings = [item["embedding"] for item in data["data"]]
    return np.array(embeddings, dtype=np.float32)  # shape (N, 1536)

# ----------------- Your original code (slightly trimmed for clarity) -----------------
PHI = (1 + 5**0.5) / 2.0
DEFAULT_N_NODES = 10000
BATCH_SIZE = 256
MAX_ITERS = 40
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def decay_time(flux_hz: float, mass: float = 1e-22, radius: float = 1e-9) -> float:
    hbar = 1.0545718e-34
    G = 6.6743e-11
    E = (4.0 * math.pi / 5.0) * G * (mass**2) / max(radius, 1e-12)
    base_tau = hbar / (E + 1e-40)
    gamma = float(flux_hz) / 500.0
    return float(base_tau / (1.0 + gamma**2))

class SignalEmbedNet(nn.Module):
    def __init__(self, embed_size: int = 64):
        super().__init__()
        self.embed_size = embed_size
        self.base_a = nn.Parameter(torch.randn(embed_size) * 0.5)
        self.base_b = nn.Parameter(torch.randn(embed_size) * 0.5)
        self.base_c = nn.Parameter(torch.randn(embed_size) * 0.5)
        self.aux_proj = nn.Sequential(
            nn.Linear(self.embed_size, self.embed_size),
            nn.ReLU(),
            nn.Linear(self.embed_size, self.embed_size)
        )

    def forward(self, flux_batch: torch.Tensor, aux_batch: torch.Tensor | None = None):
        if flux_batch.dim() == 1:
            flux_batch = flux_batch.unsqueeze(1)
        B = flux_batch.size(0)
        flux_expanded = flux_batch.repeat(1, self.embed_size)

        if aux_batch is not None:
            aux_feat = self.aux_proj(aux_batch)
            mix = 0.5 * aux_feat
        else:
            mix = torch.zeros((B, self.embed_size), device=flux_batch.device)

        a = flux_expanded * self.base_a.unsqueeze(0) + mix
        b = flux_expanded * self.base_b.unsqueeze(0) * (PHI ** 1) * 2.0 + mix
        c = flux_expanded * self.base_c.unsqueeze(0) * (PHI ** 2) + mix
        return [a, b, c]

class NodeGraphNet(nn.Module):
    def __init__(self, n_nodes: int = DEFAULT_N_NODES, embed_size: int = 64):
        super().__init__()
        self.n_nodes = n_nodes
        self.embed_size = embed_size
        self.node_embed = nn.Embedding(n_nodes, embed_size)
        self.fc = nn.Linear(embed_size * 4, 1)
        self.sigmoid = nn.Sigmoid()
        self.graph = nx.Graph()

    def forward(self, node_idx: torch.Tensor, signal_list: list[torch.Tensor]):
        node_emb = self.node_embed(node_idx.long())
        concat_signals = torch.cat(signal_list, dim=1)
        x = torch.cat([node_emb, concat_signals], dim=1)
        logits = self.fc(x)
        p_activation = self.sigmoid(logits)

        # simple dynamic edge growth (same as your original)
        if torch.mean(p_activation) > 0.5 and len(self.graph) < self.n_nodes:
            i = random.randint(0, self.n_nodes-1)
            j = random.randint(0, self.n_nodes-1)
            if i != j and not self.graph.has_edge(i, j):
                self.graph.add_edge(i, j, weight=random.uniform(0.5, 1.5))

        return p_activation

# ----------------- Main optimization with real Grok embeddings -----------------
def main():
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise SystemExit("Set XAI_API_KEY=your_key_here and rerun")

    print("Fetching real Grok embeddings...")
    reef_texts = [
        "Coral reef bleaching event detected in Great Barrier Reef",
        "Ocean temperature anomaly +1.8°C sustained for 42 days",
        "Healthy staghorn coral colony showing vibrant fluorescence",
        "Massive crown-of-thorns starfish outbreak reported",
        "Successful coral restoration using larval reseeding",
        "pH drop to 7.91 measured at reef crest",
        "Symbiotic zooxanthellae density critically low",
        "New marine protected area established in Coral Triangle"
    ] * 32  # 256 total for batching

    grok_emb = get_grok_embeddings(reef_texts, api_key)[:BATCH_SIZE*4]  # (1024, 1536)
    print(f"Received Grok embeddings: {grok_emb.shape}")

    # Project 1536 → 64 dim (random projection for speed, can be learned later)
    projector = nn.Linear(1536, 64, bias=False)
    torch.nn.init.normal_(projector.weight, std=0.02)
    aux_pool = projector(torch.from_numpy(grok_emb)).detach().numpy()

    embed_net = SignalEmbedNet(embed_size=64)
    graph_net = NodeGraphNet(n_nodes=DEFAULT_N_NODES, embed_size=64)

    optimizer = optim.Adam(list(embed_net.parameters()) + list(graph_net.parameters()), lr=3e-3)
    mse = nn.MSELoss()
    device = "cpu"

    flux_vec = np.random.uniform(40.0, 500.0, DEFAULT_N_NODES).astype(np.float32)

    print("Starting training loop with real Grok embeddings...\n")
    for it in range(MAX_ITERS):
        idx = np.random.choice(len(flux_vec), size=BATCH_SIZE, replace=False)
        flux_batch = torch.tensor(flux_vec[idx], dtype=torch.float32, device=device)
        aux_idx = np.random.choice(len(aux_pool), size=BATCH_SIZE, replace=False)
        aux_batch = torch.tensor(aux_pool[aux_idx], dtype=torch.float32, device=device)

        signal_list = embed_net(flux_batch, aux_batch)
        node_idx = torch.randint(0, DEFAULT_N_NODES, (BATCH_SIZE,), device=device)
        p_activation = graph_net(node_idx, signal_list)

        target = torch.ones_like(p_activation)
        loss = mse(p_activation, target) + 0.2 * torch.mean(p_activation)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (it+1) % 10 == 0 or it == MAX_ITERS-1:
            print(f"Iter {it+1:2d}/{MAX_ITERS}  Loss: {loss.item():.6f}  "
                  f"Mean activation: {p_activation.mean().item():.4f}  "
                  f"Edges: {graph_net.graph.number_of_edges()}")

    print("\nDone. Real Grok embeddings successfully drove the resonant graph.")

if __name__ == "__main__":
    main()