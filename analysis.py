"""Analyze the Health Search Study data.

Loads LaBrowser export data from a study of 20 participants researching
intermittent fasting. Produces summary statistics and charts.

Usage:
    python analysis.py

Output:
    - Summary statistics printed to stdout
    - Charts saved to output/ directory as PNG files
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

sns.set_theme(style="white", font_scale=1.1)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})


def load_json(filename: str) -> list | dict:
    """Load a JSON file from the data directory."""
    path = DATA_DIR / filename
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load all data files and return DataFrames + study config."""
    events = pd.DataFrame(load_json("events.json"))
    events["timestamp_utc"] = pd.to_datetime(events["timestamp_utc"])

    google_search = pd.DataFrame(load_json("google_search_v1.json"))
    google_search["start_time"] = pd.to_datetime(google_search["start_time"])
    google_search["end_time"] = pd.to_datetime(google_search["end_time"])

    chatgpt = pd.DataFrame(load_json("chatgpt_session_v1.json"))
    chatgpt["start_time"] = pd.to_datetime(chatgpt["start_time"])
    chatgpt["end_time"] = pd.to_datetime(chatgpt["end_time"])

    study_config = load_json("study_config.json")

    return events, google_search, chatgpt, study_config


# ---------------------------------------------------------------------------
# Study overview
# ---------------------------------------------------------------------------

def print_study_overview(
    events: pd.DataFrame,
    google_search: pd.DataFrame,
    chatgpt: pd.DataFrame,
    study_config: dict,
) -> pd.DataFrame:
    """Print study-level statistics. Returns session summary DataFrame."""
    participants = events["session_id"].nunique()
    total_events = len(events)
    total_searches = len(google_search)
    total_chatgpt = len(chatgpt)

    # Session durations
    session_times = events.groupby("session_id")["timestamp_utc"].agg(["min", "max"])
    session_times["duration_min"] = (
        (session_times["max"] - session_times["min"]).dt.total_seconds() / 60
    )

    # Queries per session
    queries_per_session = google_search.groupby("session_id").size()

    # ChatGPT users
    chatgpt_sessions = set(chatgpt["session_id"].unique())
    chatgpt_user_count = len(chatgpt_sessions)

    print("=" * 64)
    print("  HEALTH SEARCH STUDY - Analysis Summary")
    print("=" * 64)
    print()
    print("  Study Overview")
    print("  " + "-" * 40)
    print(f"  Participants:            {participants}")
    print(f"  Total events:            {total_events:,}")
    print(f"  Google search sessions:  {total_searches}")
    print(f"  ChatGPT conversations:   {total_chatgpt}")
    print(f"  Allowed domains:         {len(study_config['allowed_domains'])}")
    print()
    print(f"  Session duration:        {session_times['duration_min'].mean():.1f} min avg "
          f"(range: {session_times['duration_min'].min():.1f}"
          f"-{session_times['duration_min'].max():.1f})")
    print(f"  Queries per participant: {queries_per_session.mean():.1f} avg "
          f"(range: {queries_per_session.min()}-{queries_per_session.max()})")
    print(f"  ChatGPT users:           {chatgpt_user_count} "
          f"({chatgpt_user_count / participants * 100:.0f}%)")
    print()

    # Event type breakdown
    event_counts = events["event_type"].value_counts()
    print("  Event Types")
    print("  " + "-" * 40)
    for event_type, count in event_counts.items():
        print(f"  {event_type:25s} {count:>6,}")
    print()

    return session_times


# ---------------------------------------------------------------------------
# Chart: Session durations
# ---------------------------------------------------------------------------

def plot_session_durations(session_times: pd.DataFrame) -> None:
    """Histogram of session durations in minutes."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        session_times["duration_min"],
        bins=10,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
    )
    ax.set_xlabel("Session Duration (minutes)")
    ax.set_ylabel("Number of Participants")
    ax.set_title("Distribution of Session Durations")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    median_dur = session_times["duration_min"].median()
    ax.axvline(median_dur, color="#C44E52", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(
        median_dur + 0.3, ax.get_ylim()[1] * 0.9,
        f"Median: {median_dur:.1f} min",
        color="#C44E52", fontsize=10,
    )

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "session_durations.png")
    plt.close(fig)
    print("  Saved: output/session_durations.png")


# ---------------------------------------------------------------------------
# Chart: Top queries
# ---------------------------------------------------------------------------

def plot_top_queries(google_search: pd.DataFrame) -> None:
    """Horizontal bar chart of the top 10 most frequent search queries."""
    queries = google_search["payload"].apply(lambda p: p["query"])
    top_queries = queries.value_counts().head(10).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.barh(
        top_queries.index,
        top_queries.values,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.5,
        height=0.7,
    )

    ax.set_xlabel("Number of Searches")
    ax.set_title("Top 10 Search Queries")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Add count labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{int(width)}", va="center", fontsize=9, color="#333333",
        )

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_queries.png")
    plt.close(fig)
    print("  Saved: output/top_queries.png")


# ---------------------------------------------------------------------------
# Chart: Domain clicks
# ---------------------------------------------------------------------------

def extract_clicks(google_search: pd.DataFrame) -> pd.DataFrame:
    """Flatten result_clicks from all search sessions into a DataFrame."""
    rows = []
    for _, row in google_search.iterrows():
        for click in row["payload"]["result_clicks"]:
            rows.append({
                "session_id": row["session_id"],
                "query": row["payload"]["query"],
                "url": click["url"],
                "hostname": click["hostname"],
                "dwell_ms": click["dwell_ms"],
                "click_timestamp": click["click_timestamp"],
            })
    clicks_df = pd.DataFrame(rows)
    clicks_df["click_timestamp"] = pd.to_datetime(clicks_df["click_timestamp"])
    return clicks_df


def plot_domain_clicks(clicks_df: pd.DataFrame) -> None:
    """Horizontal bar chart of clicks by domain."""
    domain_clicks = clicks_df["hostname"].value_counts().head(12).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.barh(
        domain_clicks.index,
        domain_clicks.values,
        color="#55A868",
        edgecolor="white",
        linewidth=0.5,
        height=0.7,
    )

    ax.set_xlabel("Number of Clicks")
    ax.set_title("Result Clicks by Domain")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.2, bar.get_y() + bar.get_height() / 2,
            f"{int(width)}", va="center", fontsize=9, color="#333333",
        )

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "domain_clicks.png")
    plt.close(fig)
    print("  Saved: output/domain_clicks.png")


# ---------------------------------------------------------------------------
# Chart: Dwell time by domain
# ---------------------------------------------------------------------------

def plot_dwell_by_domain(clicks_df: pd.DataFrame) -> None:
    """Horizontal bar chart of average dwell time per domain."""
    # Filter to clicks that have a dwell_ms value (the last click in each
    # search sequence has dwell_ms=None because the user didn't navigate away)
    with_dwell = clicks_df[clicks_df["dwell_ms"].notna()].copy()
    with_dwell["dwell_s"] = with_dwell["dwell_ms"] / 1000

    domain_dwell = (
        with_dwell.groupby("hostname")["dwell_s"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg_dwell_s", "count": "n_clicks"})
    )
    # Only show domains with at least 3 clicks for meaningful averages
    domain_dwell = domain_dwell[domain_dwell["n_clicks"] >= 3]
    domain_dwell = domain_dwell.sort_values("avg_dwell_s")

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.barh(
        domain_dwell.index,
        domain_dwell["avg_dwell_s"],
        color="#DD8452",
        edgecolor="white",
        linewidth=0.5,
        height=0.7,
    )

    ax.set_xlabel("Average Dwell Time (seconds)")
    ax.set_title("Average Dwell Time by Domain")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, (_, row) in zip(bars, domain_dwell.iterrows()):
        width = bar.get_width()
        ax.text(
            width + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{width:.0f}s (n={int(row['n_clicks'])})",
            va="center", fontsize=9, color="#333333",
        )

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "dwell_by_domain.png")
    plt.close(fig)
    print("  Saved: output/dwell_by_domain.png")


# ---------------------------------------------------------------------------
# Chart: ChatGPT comparison
# ---------------------------------------------------------------------------

def plot_chatgpt_comparison(
    google_search: pd.DataFrame,
    chatgpt: pd.DataFrame,
    clicks_df: pd.DataFrame,
) -> None:
    """Grouped bar chart comparing ChatGPT users vs non-users."""
    chatgpt_sessions = set(chatgpt["session_id"].unique())
    all_sessions = set(google_search["session_id"].unique())
    non_chatgpt_sessions = all_sessions - chatgpt_sessions

    def session_stats(session_ids: set) -> dict:
        gs = google_search[google_search["session_id"].isin(session_ids)]
        cl = clicks_df[clicks_df["session_id"].isin(session_ids)]
        queries_per = gs.groupby("session_id").size()
        domains_per = cl.groupby("session_id")["hostname"].nunique()
        dwell_valid = cl[cl["dwell_ms"].notna()]
        avg_dwell = dwell_valid["dwell_ms"].mean() / 1000 if len(dwell_valid) > 0 else 0
        return {
            "Avg Queries": queries_per.mean() if len(queries_per) > 0 else 0,
            "Avg Unique Domains": domains_per.mean() if len(domains_per) > 0 else 0,
            "Avg Dwell (s)": avg_dwell,
        }

    chatgpt_stats = session_stats(chatgpt_sessions)
    non_chatgpt_stats = session_stats(non_chatgpt_sessions)

    metrics = list(chatgpt_stats.keys())
    chatgpt_vals = [chatgpt_stats[m] for m in metrics]
    non_chatgpt_vals = [non_chatgpt_stats[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(
        x - width / 2, non_chatgpt_vals, width,
        label=f"Google Only (n={len(non_chatgpt_sessions)})",
        color="#4C72B0", edgecolor="white", linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2, chatgpt_vals, width,
        label=f"Google + ChatGPT (n={len(chatgpt_sessions)})",
        color="#C44E52", edgecolor="white", linewidth=0.5,
    )

    ax.set_ylabel("Value")
    ax.set_title("ChatGPT Users vs Non-Users")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, height + 0.2,
                f"{height:.1f}", ha="center", va="bottom", fontsize=9,
            )

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "chatgpt_comparison.png")
    plt.close(fig)
    print("  Saved: output/chatgpt_comparison.png")


# ---------------------------------------------------------------------------
# Chart: Participant strategies
# ---------------------------------------------------------------------------

def plot_participant_strategies(
    google_search: pd.DataFrame,
    chatgpt: pd.DataFrame,
    clicks_df: pd.DataFrame,
) -> None:
    """Scatter plot of queries vs avg dwell time, colored by strategy."""
    chatgpt_sessions = set(chatgpt["session_id"].unique())

    # Build per-participant metrics
    participants = []
    for session_id in google_search["session_id"].unique():
        gs = google_search[google_search["session_id"] == session_id]
        cl = clicks_df[clicks_df["session_id"] == session_id]

        num_queries = len(gs)
        dwell_valid = cl[cl["dwell_ms"].notna()]
        avg_dwell_s = dwell_valid["dwell_ms"].mean() / 1000 if len(dwell_valid) > 0 else 0
        uses_chatgpt = session_id in chatgpt_sessions

        participants.append({
            "session_id": session_id,
            "num_queries": num_queries,
            "avg_dwell_s": avg_dwell_s,
            "uses_chatgpt": uses_chatgpt,
        })

    part_df = pd.DataFrame(participants)

    # Classify strategies
    median_queries = part_df["num_queries"].median()
    median_dwell = part_df["avg_dwell_s"].median()

    def classify(row: pd.Series) -> str:
        if row["uses_chatgpt"]:
            return "AI-Assisted"
        elif row["num_queries"] <= median_queries and row["avg_dwell_s"] >= median_dwell:
            return "Deep Diver"
        elif row["num_queries"] > median_queries and row["avg_dwell_s"] < median_dwell:
            return "Wide Scanner"
        else:
            return "Mixed"

    part_df["strategy"] = part_df.apply(classify, axis=1)

    strategy_colors = {
        "Deep Diver": "#4C72B0",
        "Wide Scanner": "#55A868",
        "AI-Assisted": "#C44E52",
        "Mixed": "#8C8C8C",
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    for strategy, color in strategy_colors.items():
        subset = part_df[part_df["strategy"] == strategy]
        if len(subset) == 0:
            continue
        ax.scatter(
            subset["num_queries"],
            subset["avg_dwell_s"],
            c=color,
            s=100,
            alpha=0.8,
            edgecolors="white",
            linewidth=0.8,
            label=f"{strategy} (n={len(subset)})",
            zorder=3,
        )

    ax.set_xlabel("Number of Search Queries")
    ax.set_ylabel("Average Dwell Time (seconds)")
    ax.set_title("Participant Research Strategies")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Light grid for readability
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "participant_strategies.png")
    plt.close(fig)
    print("  Saved: output/participant_strategies.png")


# ---------------------------------------------------------------------------
# Summary findings
# ---------------------------------------------------------------------------

def print_summary_findings(
    events: pd.DataFrame,
    google_search: pd.DataFrame,
    chatgpt: pd.DataFrame,
    clicks_df: pd.DataFrame,
) -> None:
    """Print key findings to stdout."""
    chatgpt_sessions = set(chatgpt["session_id"].unique())
    all_sessions = set(google_search["session_id"].unique())
    non_chatgpt_sessions = all_sessions - chatgpt_sessions

    # ChatGPT vs non-ChatGPT query counts
    chatgpt_queries = (
        google_search[google_search["session_id"].isin(chatgpt_sessions)]
        .groupby("session_id").size()
    )
    non_chatgpt_queries = (
        google_search[google_search["session_id"].isin(non_chatgpt_sessions)]
        .groupby("session_id").size()
    )

    # Dwell by domain (top 5)
    with_dwell = clicks_df[clicks_df["dwell_ms"].notna()].copy()
    with_dwell["dwell_s"] = with_dwell["dwell_ms"] / 1000
    top_dwell = (
        with_dwell.groupby("hostname")["dwell_s"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg_dwell_s", "count": "n"})
        .query("n >= 3")
        .sort_values("avg_dwell_s", ascending=False)
        .head(5)
    )

    print("  Key Findings")
    print("  " + "-" * 40)
    print()
    print("  1. ChatGPT users issued fewer search queries:")
    print(f"     - ChatGPT users:     {chatgpt_queries.mean():.1f} queries/session avg")
    print(f"     - Non-ChatGPT users: {non_chatgpt_queries.mean():.1f} queries/session avg")
    print()
    print("  2. Top domains by average dwell time:")
    for hostname, row in top_dwell.iterrows():
        print(f"     - {hostname:35s} {row['avg_dwell_s']:.0f}s (n={int(row['n'])})")
    print()
    print(f"  3. {len(chatgpt_sessions)} of {len(all_sessions)} participants "
          f"({len(chatgpt_sessions) / len(all_sessions) * 100:.0f}%) used ChatGPT")
    print()

    # Total prompts
    total_prompts = chatgpt["payload"].apply(lambda p: p["prompt_count"]).sum()
    print(f"  4. ChatGPT usage: {total_prompts} total prompts across "
          f"{len(chatgpt)} conversations")
    print()

    # Click-through rate (searches with at least one click)
    searches_with_clicks = google_search["payload"].apply(
        lambda p: len(p["result_clicks"]) > 0
    ).sum()
    print(f"  5. Click-through rate: {searches_with_clicks}/{len(google_search)} "
          f"({searches_with_clicks / len(google_search) * 100:.0f}%) of searches "
          f"had at least one result click")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("Loading data...")
    events, google_search, chatgpt, study_config = load_data()
    print(f"  events.json:             {len(events):,} rows")
    print(f"  google_search_v1.json:   {len(google_search):,} rows")
    print(f"  chatgpt_session_v1.json: {len(chatgpt):,} rows")
    print()

    # Overview
    session_times = print_study_overview(events, google_search, chatgpt, study_config)

    # Extract clicks for reuse
    clicks_df = extract_clicks(google_search)
    print(f"  Total result clicks:     {len(clicks_df):,}")
    print()

    # Generate charts
    print("  Generating Charts")
    print("  " + "-" * 40)
    plot_session_durations(session_times)
    plot_top_queries(google_search)
    plot_domain_clicks(clicks_df)
    plot_dwell_by_domain(clicks_df)
    plot_chatgpt_comparison(google_search, chatgpt, clicks_df)
    plot_participant_strategies(google_search, chatgpt, clicks_df)
    print()

    # Summary
    print_summary_findings(events, google_search, chatgpt, clicks_df)
    print("=" * 64)
    print("  Analysis complete. Charts saved to output/")
    print("=" * 64)
    print()


if __name__ == "__main__":
    main()
