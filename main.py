"""Generate synthetic e-commerce data and save it as Parquet files."""

import argparse
import logging
import time
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.config import settings as config
from src.data_generator import SyntheticDataGenerator


LOGGER = logging.getLogger(__name__)


def format_size(file_size: int) -> str:
    """Format a file size in bytes as a readable value."""
    size = float(file_size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{file_size} B"


def save_parquet(dataframe: pd.DataFrame, output_path: Path) -> int:
    """Save a pandas DataFrame as compressed Parquet and return its file size."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Spark reads millisecond timestamps reliably across supported Spark versions.
    dataframe.to_parquet(
        output_path,
        index=False,
        compression="snappy",
        coerce_timestamps="ms",
        allow_truncated_timestamps=True,
    )
    return output_path.stat().st_size


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse optional record counts and the random seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=100_000)
    parser.add_argument("--products", type=int, default=10_000)
    parser.add_argument("--orders", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Generate datasets, save Parquet files, and print timing and size details."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(arguments)
    start_time = time.perf_counter()

    try:
        generator = SyntheticDataGenerator(seed=args.seed)
        customers, products, orders = generator.generate_all(
            customer_count=args.customers,
            product_count=args.products,
            order_count=args.orders,
        )

        output_files = {
            "customers": config.raw_data_dir / "customers.parquet",
            "products": config.raw_data_dir / "products.parquet",
            "orders": config.raw_data_dir / "orders.parquet",
        }
        dataframes = {"customers": customers, "products": products, "orders": orders}
        print("Generated files:")
        for name, dataframe in dataframes.items():
            file_size = save_parquet(dataframe, output_files[name])
            print(f"  {output_files[name]}: {format_size(file_size)}")

        elapsed = time.perf_counter() - start_time
        print(f"Generation time: {elapsed:.2f} seconds")
        return 0
    except Exception:
        LOGGER.exception("E-commerce data generation failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())