from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_price(price: pd.Series, critical: pd.Series, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(price.index, price.values, label="EUETS Price", color="#1f77b4")
    critical_idx = critical[critical == 1].index
    ax.scatter(critical_idx, price.loc[critical_idx], color="#d62728", label="Critical", s=10)
    ax.set_title("EUETS Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    fig.tight_layout()
    pdf_path = output_dir / "EUETS_Price_Over_Time.pdf"
    png_path = output_dir / "EUETS_Price_Over_Time.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def plot_risk_curve(scores: pd.Series, target: pd.Series, output_dir: Path, w: int, tau: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(scores.index, scores.values, label="Risk Score", color="#ff7f0e")
    ax.fill_between(target.index, 0, target.values, color="#1f77b4", alpha=0.2, label="Future Critical")
    ax.set_title(f"Risk Curve (w={w}, tau={tau})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Risk")
    ax.legend(loc="best")
    fig.tight_layout()
    pdf_path = output_dir / f"Fig_risk_curve_w{w}_tau{tau}.pdf"
    png_path = output_dir / f"Fig_risk_curve_w{w}_tau{tau}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
