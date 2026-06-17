from pathlib import Path
import re
import sys
import pandas as pd


def parse_fps_from_csv(df: pd.DataFrame, default_fps: float = 30.0) -> float:
    """
    Try to read fps from the row where check_name == 'check_fps'.
    Example reason: 'fps=30.00 >= 5'
    """
    fps_rows = df[df["check_name"].eq("check_fps")]

    if fps_rows.empty:
        return default_fps

    reason = str(fps_rows.iloc[0]["reason"])
    match = re.search(r"fps=([0-9.]+)", reason)

    if not match:
        return default_fps

    return float(match.group(1))


def seconds_to_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    sec = seconds % 60
    return f"{minutes:02d}:{sec:05.2f}"


def print_csv_result(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)

    fps = parse_fps_from_csv(df)

    print("=" * 100)
    print(f"CSV: {csv_path.name}")
    print(f"Video filename: {df['filename'].iloc[0]}")
    print(f"Volunteer ID: {df['volunteer_id'].iloc[0]}")
    print(f"Detected FPS: {fps}")
    print(f"Total rows: {len(df)}")
    print()

    # 1) Overall status count
    print("[Overall status count]")
    print(df["status"].value_counts().to_string())
    print()

    # 2) Check-by-check summary
    print("[Check summary]")
    summary = (
        df.pivot_table(
            index="check_name",
            columns="status",
            values="reason",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )
    print(summary.to_string(index=False))
    print()

    # 3) FAIL / REVIEW rows
    problem_df = df[df["status"].isin(["FAIL", "REVIEW"])].copy()

    if problem_df.empty:
        print("[Problems]")
        print("No FAIL or REVIEW rows.")
        print()
        return

    problem_df["frame_index"] = problem_df["frame_index"].astype("Int64")

    problem_df["timestamp_sec"] = problem_df["frame_index"].apply(
        lambda x: None if pd.isna(x) else x / fps
    )

    problem_df["timestamp"] = problem_df["timestamp_sec"].apply(
        lambda x: "" if pd.isna(x) else seconds_to_timestamp(x)
    )

    print("[FAIL / REVIEW rows]")
    cols = [
        "status",
        "check_name",
        "frame_index",
        "timestamp",
        "reason",
    ]
    print(problem_df[cols].to_string(index=False))
    print()


def main():
    # Usage:
    #   python print_csv_results.py
    #   python print_csv_results.py face_002.csv
    #   python print_csv_results.py results_folder
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    if target.is_file() and target.suffix.lower() == ".csv":
        csv_files = [target]
    elif target.is_dir():
        csv_files = sorted(target.glob("*.csv"))
    else:
        raise FileNotFoundError(f"Cannot find CSV file/folder: {target}")

    if not csv_files:
        print(f"No CSV files found in: {target}")
        return

    for csv_path in csv_files:
        print_csv_result(csv_path)


if __name__ == "__main__":
    main()