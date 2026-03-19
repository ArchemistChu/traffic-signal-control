#!/usr/bin/env python3
"""
Generate training-only figures from a MARL checkpoint.

This script is useful when evaluation JSON is not ready yet but the checkpoint
already contains training history. It produces the main figures used in the
report: reward, loss, epsilon, dashboard, and convergence analysis. If the
checkpoint contains the newer `episode_switch_rates` field, a switch-rate plot
is also generated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


COLOR_RAW = "#C8D0D9"
COLOR_REWARD = "#C1121F"
COLOR_LOSS = "#1D3557"
COLOR_EPS = "#2A9D8F"
COLOR_LEN = "#F4A261"
COLOR_SWITCH = "#6A4C93"


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window=window, min_periods=1).mean().to_numpy()


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(window=window, min_periods=max(5, window // 4))
        .std()
        .bfill()
        .ffill()
        .to_numpy()
    )


def approximate_episode_series(step_values: np.ndarray, n_episodes: int) -> np.ndarray:
    if len(step_values) == 0 or n_episodes <= 0:
        return np.array([], dtype=float)
    edges = np.linspace(0, len(step_values), n_episodes + 1, dtype=int)
    episode_means = []
    for start, end in zip(edges[:-1], edges[1:]):
        chunk = step_values[start:end]
        episode_means.append(float(np.mean(chunk)) if len(chunk) else np.nan)
    return np.array(episode_means, dtype=float)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def save_figure(fig: plt.Figure, stem: str, output_roots: Iterable[Path]) -> None:
    for root in output_roots:
        root.mkdir(parents=True, exist_ok=True)
        fig.savefig(root / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(root / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_series_plot(
    ax: plt.Axes,
    episodes: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    ylabel: str,
    raw_color: str,
    smooth_color: str,
    smooth_window: int = 10,
    show_std_band: bool = True,
) -> None:
    ax.plot(episodes, values, color=raw_color, linewidth=0.6, alpha=0.9, label="Per-episode")
    smooth = moving_average(values, smooth_window)
    ax.plot(episodes, smooth, color=smooth_color, linewidth=2.0, label=f"MA-{smooth_window}")
    if show_std_band and len(values) >= 5:
        std = rolling_std(values, 20)
        ax.fill_between(
            episodes,
            smooth - std,
            smooth + std,
            color=smooth_color,
            alpha=0.15,
            linewidth=0.0,
            label="Rolling std.",
        )
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="models/marl_cologne_shared_dqn_regionaware_v9_scale07.pt",
        help="Path to the trained checkpoint (.pt)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="Report/figures/training",
        help="Primary output directory for training figures",
    )
    args = parser.parse_args()

    root = Path(".").resolve()
    model_path = (root / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    out_dir = (root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    report_fig_dir = root / "Report" / "figures"
    output_roots = [
        out_dir,
        out_dir / "png",
        out_dir / "pdf",
        report_fig_dir,
    ]

    setup_style()

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    history = checkpoint.get("training_history", {})
    config = checkpoint.get("config", {})

    reward_mean = np.array(
        history.get("episode_reward_mean_per_tls_step", history.get("episode_rewards", [])),
        dtype=float,
    )
    reward_sum = np.array(history.get("episode_reward_sums", []), dtype=float)
    episode_lengths = np.array(history.get("episode_lengths", []), dtype=float)
    losses = np.array(history.get("losses", []), dtype=float)
    epsilon_values = np.array(history.get("epsilon_values", []), dtype=float)
    switch_rates = np.array(history.get("episode_switch_rates", []), dtype=float)

    n_episodes = len(reward_mean)
    if n_episodes == 0:
        raise RuntimeError(f"No episode-level reward history found in checkpoint: {model_path}")

    episodes = np.arange(1, n_episodes + 1, dtype=int)
    episode_losses = approximate_episode_series(losses, n_episodes)
    episode_epsilon = approximate_episode_series(epsilon_values, n_episodes)

    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Action dim: {config.get('action_dim', 'N/A')}")
    print(f"Decision interval: {config.get('decision_interval', 'N/A')}")
    print(f"Controlled lights ratio: {config.get('controlled_lights_ratio', 'N/A')}")
    print(f"Final epsilon: {checkpoint.get('epsilon', 'N/A')}")
    print(f"Reward mean/tl-step last 20: {np.nanmean(reward_mean[-20:]):.4f}")
    if len(reward_sum):
        print(f"Reward sum last 20: {np.nanmean(reward_sum[-20:]):.2f}")
    if len(episode_losses):
        print(f"Approx. episode loss last 20: {np.nanmean(episode_losses[-20:]):.4f}")
    if len(switch_rates):
        print(f"Switch rate last 20: {np.nanmean(switch_rates[-20:]):.4f}")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    add_series_plot(
        ax,
        episodes,
        reward_mean,
        title="Training Reward per TLS-Step",
        ylabel="Reward",
        raw_color=COLOR_RAW,
        smooth_color=COLOR_REWARD,
    )
    ax.legend(loc="best")
    save_figure(fig, "training_reward_curve", output_roots)

    if len(episode_losses):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        add_series_plot(
            ax,
            episodes,
            episode_losses,
            title="Training Loss per Episode",
            ylabel="Smooth L1 Loss",
            raw_color=COLOR_RAW,
            smooth_color=COLOR_LOSS,
        )
        ax.legend(loc="best")
        save_figure(fig, "training_loss_curve", output_roots)

    if len(episode_epsilon):
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.plot(episodes, episode_epsilon, color=COLOR_EPS, linewidth=1.8)
        ax.set_title("Exploration Schedule")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon")
        ax.grid(True, alpha=0.3)
        save_figure(fig, "training_epsilon_decay", output_roots)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    add_series_plot(
        axes[0, 0],
        episodes,
        reward_mean,
        title="(a) Reward",
        ylabel="Reward",
        raw_color=COLOR_RAW,
        smooth_color=COLOR_REWARD,
    )
    if len(episode_losses):
        add_series_plot(
            axes[0, 1],
            episodes,
            episode_losses,
            title="(b) Loss",
            ylabel="Smooth L1 Loss",
            raw_color=COLOR_RAW,
            smooth_color=COLOR_LOSS,
        )
    else:
        axes[0, 1].set_title("(b) Loss")
        axes[0, 1].text(0.5, 0.5, "No loss history", ha="center", va="center")
    if len(episode_epsilon):
        axes[1, 0].plot(episodes, episode_epsilon, color=COLOR_EPS, linewidth=1.8)
    axes[1, 0].set_title("(c) Epsilon")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Epsilon")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(episodes, episode_lengths, color=COLOR_LEN, linewidth=1.6)
    axes[1, 1].set_title("(d) Episode Length")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Stored transitions")
    axes[1, 1].grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "training_dashboard_4panel", output_roots)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    axes[0].plot(episodes, moving_average(reward_mean, 10), color=COLOR_REWARD, linewidth=2.0)
    axes[0].set_title("Reward Moving Average")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(episodes, pd.Series(reward_mean).expanding().mean().to_numpy(), color=COLOR_LOSS, linewidth=2.0)
    axes[1].set_title("Cumulative Mean Reward")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Reward")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(episodes, rolling_std(reward_mean, 20), color=COLOR_EPS, linewidth=2.0)
    axes[2].set_title("Rolling Reward Std.")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Std. Dev.")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "training_convergence_analysis", output_roots)

    if len(switch_rates):
        fig, ax = plt.subplots(figsize=(7, 3.8))
        add_series_plot(
            ax,
            episodes,
            switch_rates,
            title="Training Phase Switch Rate",
            ylabel="Switch rate",
            raw_color=COLOR_RAW,
            smooth_color=COLOR_SWITCH,
            show_std_band=False,
        )
        ax.legend(loc="best")
        save_figure(fig, "training_switch_rate_curve", output_roots)

    if len(reward_sum):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        add_series_plot(
            ax,
            episodes,
            reward_sum,
            title="Training Reward Sum per Episode",
            ylabel="Reward sum",
            raw_color=COLOR_RAW,
            smooth_color=COLOR_REWARD,
        )
        ax.legend(loc="best")
        save_figure(fig, "training_reward_sum_curve", output_roots)

    print("Saved training figures to:")
    for root_dir in output_roots:
        print(f"  {root_dir}")


if __name__ == "__main__":
    main()
