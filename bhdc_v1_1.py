"""BHDC v1.1 - corrected coherent core (collapse last; one shared dilation operator;
branches = scale x geometry; unitary geometry bank so the coherent sum is well-defined)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch, torch.nn as nn, torch.nn.functional as F


def complex_overlap(a, b):
    return torch.sum(a.conj() * b, dim=-1)


def collision_density(psi):
    p = (psi.conj() * psi).real
    d = psi.shape[-1]
    D = (p * p).sum(dim=-1)
    return (D - 1.0 / d) / (1.0 - 1.0 / d + 1e-9)


class ComplexEncoder(nn.Module):
    def __init__(self, in_dim, d_model):
        super().__init__()
        self.to_real = nn.Linear(in_dim, d_model)
        self.to_imag = nn.Linear(in_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        psi = torch.complex(self.norm(self.to_real(x)), self.to_imag(x))
        return psi / torch.linalg.norm(psi, dim=-1, keepdim=True).clamp_min(1e-6)


class DilationSpine(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.log_omega = nn.Parameter(torch.linspace(-2.0, 2.0, d_model))
        self.B = nn.Parameter(torch.randn(d_model, 2) * 0.02)
        self.C = nn.Parameter(torch.randn(d_model, 2) * 0.02)

    def omega(self):
        return F.softplus(self.log_omega)

    def scan(self, psi, scale):
        omega = self.omega()
        lam = torch.complex(-0.5 * torch.ones_like(omega), omega)
        A = torch.exp(lam.view(1, 1, -1) * scale)
        Bc = torch.view_as_complex(self.B.contiguous())
        M, T, d = psi.shape
        st = torch.zeros(M, d, dtype=torch.cfloat, device=psi.device)
        outs = []
        for t in range(T):
            st = A[:, t, :] * st + Bc * psi[:, t, :]
            outs.append(st)
        return torch.stack(outs, dim=1)

    def readout(self, h):
        return h * torch.view_as_complex(self.C.contiguous())

    def modular_energy(self, H):
        w = self.omega().view(*([1] * (H.dim() - 1)), -1)
        return (w * (H.conj() * H).real).sum(dim=-1)


class IdentityGeometry(nn.Module):
    def forward(self, psi):
        return psi


class TorusGeometry(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.phase = nn.Parameter(torch.randn(d_model) * 0.01)

    def forward(self, psi):
        return psi * torch.exp(1j * self.phase)


class SU2BlockGeometry(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 2 == 0
        self.theta = nn.Parameter(torch.randn(d_model // 2) * 0.01)
        self.phi = nn.Parameter(torch.randn(d_model // 2) * 0.01)

    def forward(self, psi):
        M, T, d = psi.shape
        x = psi.view(M, T, d // 2, 2)
        a, b = x[..., 0], x[..., 1]
        c = torch.cos(self.theta).view(1, 1, -1)
        s = torch.sin(self.theta).view(1, 1, -1)
        e = torch.exp(1j * self.phi).view(1, 1, -1)
        na = c * a - e * s * b
        nb = e.conj() * s * a + c * b
        return torch.stack([na, nb], dim=-1).reshape(M, T, d)


class MeshGeometry(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layer1 = SU2BlockGeometry(d_model)
        self.layer2 = SU2BlockGeometry(d_model)

    def forward(self, psi):
        h = self.layer1(psi)
        h = torch.roll(h, 1, dims=-1)
        h = self.layer2(h)
        return torch.roll(h, -1, dims=-1)


def build_geometry_bank(d_model, n_branches):
    pool = [IdentityGeometry(), TorusGeometry(d_model),
            SU2BlockGeometry(d_model), MeshGeometry(d_model)]
    geoms = [pool[0]]
    i = 1
    while len(geoms) < n_branches:
        geoms.append(pool[1 + ((i - 1) % (len(pool) - 1))]); i += 1
    return nn.ModuleList(geoms[:n_branches])


class GLUMix(nn.Module):
    def __init__(self, d_model, expansion=2):
        super().__init__()
        hidden = expansion * d_model
        self.up = nn.Linear(d_model, hidden * 2)
        self.down = nn.Linear(hidden, d_model)

    def forward(self, x):
        a, gate = self.up(x).chunk(2, dim=-1)
        return self.down(a * F.silu(gate))
print("wrote bhdc_v1_1.py")
