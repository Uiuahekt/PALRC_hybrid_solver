import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


RELEASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = RELEASE_DIR / "data" / "all_method_trajectories.nc"
DEFAULT_OUTPUT_DIR = RELEASE_DIR / "output"


def display_name(name: str) -> str:
    return "Ours" if name == "Learned_Correction" else name


def sort_key(name: str):
    if name == "Learned_Correction" or name == "Ours":
        return (0, name)
    if name == "LI":
        return (1, name)
    if "baseline" in name.lower():
        size = int(name.split("_", 1)[1].split("x", 1)[0])
        return (2, size)
    return (3, name)


def compute_absolute_errors(trajectories: xr.Dataset, reference_method: str) -> xr.Dataset:
    ref = trajectories.sel({"method": reference_method})
    diff_u = trajectories["u"] - ref["u"]
    diff_v = trajectories["v"] - ref["v"]
    l2 = np.sqrt((diff_u**2 + diff_v**2).mean(dim=["x", "y"]))
    linf = xr.apply_ufunc(
        np.maximum,
        np.abs(diff_u).max(dim=["x", "y"]),
        np.abs(diff_v).max(dim=["x", "y"]),
    )
    return xr.Dataset({"l2": l2.astype(np.float32), "linf": linf.astype(np.float32)})


def summarize_errors(errors: xr.Dataset) -> xr.Dataset:
    return xr.Dataset(
        {
            "l2_mean": errors["l2"].mean(dim="seed"),
            "l2_std": errors["l2"].std(dim="seed"),
            "linf_mean": errors["linf"].mean(dim="seed"),
            "linf_std": errors["linf"].std(dim="seed"),
        }
    )


def save_overall_statistics(summary: xr.Dataset, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "mean_l2", "std_l2", "mean_linf", "std_linf"])
        for method in sorted(summary.method.values.tolist(), key=sort_key):
            item = summary.sel({"method": method})
            writer.writerow(
                [
                    display_name(str(method)),
                    float(item["l2_mean"].mean()),
                    float(item["l2_std"].mean()),
                    float(item["linf_mean"].mean()),
                    float(item["linf_std"].mean()),
                ]
            )


def save_stage_statistics(summary: xr.Dataset, output_path: Path) -> None:
    def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(values[mask].mean())

    time_vals = summary.time.values
    stages = {
        "0-4s": (time_vals >= 0) & (time_vals <= 4),
        "4-8s": (time_vals > 4) & (time_vals <= 8),
    }
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "stage", "mean_l2", "mean_linf"])
        for method in sorted(summary.method.values.tolist(), key=sort_key):
            item = summary.sel({"method": method})
            for stage, mask in stages.items():
                writer.writerow(
                    [
                        display_name(str(method)),
                        stage,
                        masked_mean(item["l2_mean"].values, mask),
                        masked_mean(item["linf_mean"].values, mask),
                    ]
                )


def plot_absolute_errors(summary: xr.Dataset, output_dir: Path, show: bool) -> None:
    methods = sorted(summary.method.values.tolist(), key=sort_key)
    time_vals = summary.time.values
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    markers = ["o", "s", "^", "D", "v", "p", "*", "X", "h", "<", ">"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, method in enumerate(methods):
        item = summary.sel({"method": method})
        marker_every = max(1, len(time_vals) // 10)
        label = display_name(str(method))
        axes[0].plot(
            time_vals,
            item["l2_mean"],
            color=colors[idx],
            marker=markers[idx % len(markers)],
            markevery=marker_every,
            markersize=5,
            linewidth=1.8,
            label=label,
        )
        axes[1].plot(
            time_vals,
            item["linf_mean"],
            color=colors[idx],
            marker=markers[idx % len(markers)],
            markevery=marker_every,
            markersize=5,
            linewidth=1.8,
            label=label,
        )

    for axis, ylabel in zip(axes, [r"$L_2$ Error", r"$L_\infty$ Error"]):
        axis.set_xlabel("Time", fontsize=12)
        axis.set_ylabel(ylabel, fontsize=12)
        axis.legend(fontsize=8)
        axis.grid(True, alpha=0.3)
        axis.set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_dir / "abs_error_vs_time_base.eps", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "abs_error_vs_time_base.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def analyze(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = xr.open_dataset(args.trajectory_nc)

    errors = compute_absolute_errors(trajectories, args.reference_method)
    errors.attrs.update(
        description="Absolute L2 and Linf errors computed from merged method trajectories.",
        reference_method=args.reference_method,
    )
    errors_path = output_dir / "absolute_error_all_methods_from_trajectories.nc"
    errors.to_netcdf(errors_path)

    summary = summarize_errors(errors)
    summary_path = output_dir / "absolute_error_summary.nc"
    summary.to_netcdf(summary_path)

    save_overall_statistics(summary, output_dir / "absolute_error_overall_statistics.csv")
    save_stage_statistics(summary, output_dir / "absolute_error_stage_statistics.csv")
    plot_absolute_errors(summary, output_dir, args.show)

    print(f"Saved: {errors_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved plots/statistics to: {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze merged all-method trajectories.")
    parser.add_argument(
        "--trajectory-nc",
        default=str(DEFAULT_DATA_PATH),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--reference-method", default="baseline_2048x2048")
    parser.add_argument("--show", action="store_true")
    return parser


if __name__ == "__main__":
    analyze(build_arg_parser().parse_args())
