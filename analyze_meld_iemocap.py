import argparse
from typing import List, Optional

import pandas as pd


def _order_splits(existing: List[str]) -> List[str]:
    preferred = ["train", "dev", "test"]
    # Keep preferred order if present, then append any others
    out = [s for s in preferred if s in existing]
    out += [s for s in existing if s not in out]
    return out


def print_split_sizes(df: pd.DataFrame, name: str) -> None:
    print(f"\n== {name}: Split sizes ==")
    split_counts = df["split"].value_counts()
    order = _order_splits(split_counts.index.tolist())
    split_counts = split_counts.reindex(order)
    total = int(split_counts.sum())
    for split, cnt in split_counts.items():
        pct = 100.0 * cnt / total if total else 0.0
        print(f"{split:>5}: {cnt:>6} ({pct:5.1f}%)")


def print_emotion_distribution_by_split(df: pd.DataFrame, name: str) -> None:
    print(f"\n== {name}: Emotion distribution by split ==")
    if not {"split", "emotion"}.issubset(df.columns):
        print("Required columns 'split' and 'emotion' not found. Skipping.")
        return

    # Crosstab counts
    counts = pd.crosstab(df["emotion"], df["split"]).sort_index()
    counts = counts.reindex(columns=_order_splits(list(counts.columns)))

    # Percentages per split
    col_sums = counts.sum(axis=0)
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print("Counts:")
        print(counts)
        print("\nPercentages (column-wise):")
        pct = counts.divide(col_sums, axis=1).fillna(0.0) * 100.0
        print(pct.round(1))


def print_speaker_distribution(
    df: pd.DataFrame, name: str, top_n: Optional[int] = 20
) -> None:
    print(f"\n== {name}: Speaker distribution across splits ==")
    if not {"split", "speaker"}.issubset(df.columns):
        print("Required columns 'split' and 'speaker' not found. Skipping.")
        return

    # Overall speaker counts
    overall = df["speaker"].value_counts()
    # Per-split distribution
    by_split = pd.crosstab(df["speaker"], df["split"])  # speakers x splits
    by_split = by_split.reindex(columns=_order_splits(list(by_split.columns)))
    # Sort speakers by overall frequency
    by_split["__total__"] = by_split.sum(axis=1)
    by_split = by_split.sort_values("__total__", ascending=False)

    if top_n is not None and top_n > 0:
        display = by_split.head(top_n)
        print(f"Top {top_n} speakers by frequency (counts per split):")
    else:
        display = by_split
        print("All speakers (counts per split):")

    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(display.drop(columns=["__total__"]))

    print("\nOverall speaker counts (top first):")
    if top_n is not None and top_n > 0:
        print(overall.head(top_n))
    else:
        with pd.option_context("display.max_rows", None):
            print(overall)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze MELD and IEMOCAP emotion/speaker distributions."
    )
    parser.add_argument(
        "--iemocap",
        default="iemocap_erc.csv",
        help="Path to IEMOCAP CSV (default: iemocap_erc.csv)",
    )
    parser.add_argument(
        "--meld",
        default="meld_erc.csv",
        help="Path to MELD CSV (default: meld_erc.csv)",
    )
    parser.add_argument(
        "--top-speakers",
        type=int,
        default=20,
        help="How many top speakers to show per dataset (set <=0 for all).",
    )

    iemocap = "iemocap_erc.csv"
    meld = "meld_erc.csv"
    top_speakers = 20

    # args = parser.parse_args()

    # Load datasets
    iemocap_df = pd.read_csv(iemocap)
    meld_df = pd.read_csv(meld)

    # IEMOCAP
    print_split_sizes(iemocap_df, "IEMOCAP")
    print_emotion_distribution_by_split(iemocap_df, "IEMOCAP")
    print_speaker_distribution(iemocap_df, "IEMOCAP", top_n=top_speakers)

    # MELD
    print_split_sizes(meld_df, "MELD")
    print_emotion_distribution_by_split(meld_df, "MELD")
    print_speaker_distribution(meld_df, "MELD", top_n=top_speakers)


if __name__ == "__main__":
    main()




# Permanently changes the pandas settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)
pd.set_option('display.precision', 2)


iemocap = "iemocap_erc.csv"
iemocap_df = pd.read_csv(iemocap)
iemocap_df["emotion"].value_counts()

iemocap_df.loc[iemocap_df["erc_target"]]["split"].value_counts()



meld = "meld_erc.csv"
meld_df = pd.read_csv(meld)
meld_df["emotion"].value_counts()
meld_df["speaker"].value_counts()
iemocap_df.loc[iemocap_df["erc_target"]]["split"].value_counts()

iemocap_df_filtered = iemocap_df[iemocap_df["erc_target"]][["split", "emotion_code", "emotion"]]

meld_df_filtered = meld_df[["split", "emotion"]]

iemocap_df_filtered["emotion"].unique()
meld_df_filtered["emotion"].unique()


iemocap_emotion_map = {
    "neutral": "neutral",
    "frustrated": "frustrated",
    "angry": "angry",
    "sad": "sad",
    "happy": "joyful",
    "excited": "excited",
}

meld_emotion_map = {
    "neutral": "neutral",
    "surprise": "surprised",
    "fear": "fearful",
    "sadness": "sad",
    "joy": "joyful",
    "disgust": "disgusted",
    "anger": "angry",
}


iemocap_df_filtered["emotion_mapped"] = iemocap_df_filtered["emotion"].map(iemocap_emotion_map)
meld_df_filtered["emotion_mapped"] = meld_df_filtered["emotion"].map(meld_emotion_map)





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")   # or "Qt5Agg", depending on what’s available


def stacked_label_distribution(datasets, order=None, title="Label Distribution Across Datasets"):
    """
    Plot stacked counts of emotion labels for multiple datasets.

    Parameters
    ----------
    datasets : dict[str, pd.Series | pd.DataFrame]
        Mapping of dataset name -> Series of labels, or DataFrame that
        contains a column named 'emotion_mapped'.
        Example: {"IEMOCAP": df_iemocap, "MELD": df_meld}
    order : list[str] | None
        Optional fixed order of labels on the x-axis. If None, labels are
        ordered by total count (descending).
    title : str
        Plot title.
    """
    # Extract Series of labels
    label_series = {}
    for name, obj in datasets.items():
        if isinstance(obj, pd.Series):
            label_series[name] = obj
        else:
            label_series[name] = obj["emotion_mapped"]

    # All labels present across datasets
    all_labels = sorted(set().union(*[s.unique() for s in label_series.values()]))

    # Count table: rows=labels, cols=datasets
    counts = pd.DataFrame(
        {name: s.value_counts().reindex(all_labels, fill_value=0)
         for name, s in label_series.items()}
    )

    # Order labels
    if order is None:
        order = counts.sum(axis=1).sort_values(ascending=False).index.tolist()
    counts = counts.loc[order]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    x = np.arange(len(counts))
    bottom = np.zeros(len(counts))

    bars_by_ds = {}
    for ds_name in counts.columns:
        vals = counts[ds_name].values
        bars = ax.bar(x, vals, bottom=bottom, label=ds_name)
        bars_by_ds[ds_name] = bars
        bottom += vals

    # Add percentage annotations per segment
    totals = counts.sum(axis=1).values
    for j, ds_name in enumerate(counts.columns):
        vals = counts[ds_name].values
        cum_before = (counts.iloc[:, :j].sum(axis=1).values if j > 0 else np.zeros_like(vals))
        for i, (v, t, b) in enumerate(zip(vals, totals, cum_before)):
            if v > 0 and t > 0:
                pct = 100.0 * v / t
                ax.text(
                    x[i], b + v / 2.0, f"{pct:.1f}%", ha="center", va="center",
                    fontsize=8, rotation=0
                )

    # Cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index, rotation=20, ha="right")
    ax.set_ylabel("Quantity")
    ax.set_xlabel("Label")
    ax.set_title(title)
    ax.legend(title="Dataset")
    ax.margins(x=0.02)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout()
    return fig, ax

# --- Usage with your two DataFrames ---
# df_iemocap and df_meld each contain a column 'emotion_mapped'
fig, ax = stacked_label_distribution(
    {
        "IEMOCAP": iemocap_df_filtered,   # or df_iemocap['emotion_mapped']
        "MELD": meld_df_filtered,         # or df_meld['emotion_mapped']
    },
    # Optionally enforce a specific label order to match your paper/figures:
    # order=["neutral", "joyful", "mad", "fear", "sad", "powerful", "excited", "peaceful", "disgust"]
    title="Label Distribution Across Datasets"
)

# Save the figure
fig.savefig("label_distribution.png", dpi=300, bbox_inches="tight")   # PNG
fig.savefig("label_distribution.pdf", bbox_inches="tight")            # PDF

plt.show()
