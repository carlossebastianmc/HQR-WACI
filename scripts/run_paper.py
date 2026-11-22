"""Fit the models and generate all tables and figures."""

from pathlib import Path

from hqr_waci.paper import run_paper


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    run_paper(
        ROOT / "data" / "df_2024.csv",
        ROOT,
    )


if __name__ == "__main__":
    main()
