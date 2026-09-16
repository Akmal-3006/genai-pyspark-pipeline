"""Compare Pandas and PySpark performance for e-commerce order analytics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import settings
from src.spark_analytics import create_spark_session


@dataclass(frozen=True)
class BenchmarkResult:
    """Store the elapsed time for one benchmark operation."""

    engine: str
    operation: str
    seconds: float


def elapsed(operation: Callable[[], Any]) -> tuple[Any, float]:
    """Run an operation and return its result with elapsed wall-clock seconds."""
    started = time.perf_counter()
    result = operation()
    return result, time.perf_counter() - started


def benchmark_pandas(raw_data_dir: Path) -> tuple[list[BenchmarkResult], pd.DataFrame]:
    """Benchmark Parquet loading, joining, revenue calculation, and aggregation in Pandas."""
    results: list[BenchmarkResult] = []

    (orders, products), seconds = elapsed(
        lambda: (
            pd.read_parquet(raw_data_dir / "orders.parquet"),
            pd.read_parquet(raw_data_dir / "products.parquet"),
        )
    )
    results.append(BenchmarkResult("Pandas", "load Parquet files", seconds))

    joined, seconds = elapsed(lambda: orders.merge(products, on="product_id", how="inner"))
    results.append(BenchmarkResult("Pandas", "join on product_id", seconds))

    with_revenue, seconds = elapsed(
        lambda: joined.assign(revenue=joined["quantity"] * joined["price"])
    )
    results.append(BenchmarkResult("Pandas", "calculate revenue", seconds))

    grouped, seconds = elapsed(
        lambda: with_revenue.groupby("customer_id", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
    )
    results.append(BenchmarkResult("Pandas", "group by customer_id", seconds))

    top_customers, seconds = elapsed(lambda: grouped.head(10))
    results.append(BenchmarkResult("Pandas", "get top 10", seconds))
    return results, top_customers


def benchmark_pyspark(
    spark: SparkSession, raw_data_dir: Path
) -> tuple[list[BenchmarkResult], DataFrame]:
    """Benchmark the equivalent operations in PySpark with lazy stages materialized."""
    results: list[BenchmarkResult] = []

    (orders, products), seconds = elapsed(
        lambda: (
            spark.read.parquet(str(raw_data_dir / "orders.parquet")),
            spark.read.parquet(str(raw_data_dir / "products.parquet")),
        )
    )
    # Spark reads are lazy, so count both inputs before stopping the load timer.
    _, materialize_seconds = elapsed(lambda: (orders.count(), products.count()))
    results.append(BenchmarkResult("PySpark", "load Parquet files", seconds + materialize_seconds))

    joined = orders.join(products, on="product_id", how="inner").persist(StorageLevel.MEMORY_AND_DISK)
    _, seconds = elapsed(joined.count)
    results.append(BenchmarkResult("PySpark", "join on product_id", seconds))

    with_revenue = joined.withColumn("revenue", F.col("quantity") * F.col("price")).persist(
        StorageLevel.MEMORY_AND_DISK
    )
    _, seconds = elapsed(with_revenue.count)
    results.append(BenchmarkResult("PySpark", "calculate revenue", seconds))

    grouped = with_revenue.groupBy("customer_id").agg(F.sum("revenue").alias("revenue")).persist(
        StorageLevel.MEMORY_AND_DISK
    )
    _, seconds = elapsed(grouped.count)
    results.append(BenchmarkResult("PySpark", "group by customer_id", seconds))

    top_customers = grouped.orderBy(F.desc("revenue")).limit(10)
    _, seconds = elapsed(top_customers.collect)
    results.append(BenchmarkResult("PySpark", "get top 10", seconds))

    joined.unpersist()
    with_revenue.unpersist()
    grouped.unpersist()
    return results, top_customers


def print_comparison(results: list[BenchmarkResult]) -> None:
    """Print benchmark results as a comparison table."""
    comparison = pd.DataFrame([result.__dict__ for result in results])
    comparison["seconds"] = comparison["seconds"].round(4)
    print("\nPerformance comparison:")
    print(comparison.to_string(index=False))

    pivot = comparison.pivot(index="operation", columns="engine", values="seconds")
    print("\nSeconds by operation:")
    print(pivot.round(4).to_string())


def main() -> None:
    """Run both benchmarks and print their top-customer results and timings."""
    spark = create_spark_session()
    try:
        pandas_results, pandas_top = benchmark_pandas(settings.raw_data_dir)
        pyspark_results, pyspark_top = benchmark_pyspark(spark, settings.raw_data_dir)
        print("\nPandas top 10 customers:")
        print(pandas_top.to_string(index=False))
        print("\nPySpark top 10 customers:")
        pyspark_top.show(truncate=False)
        print_comparison(pandas_results + pyspark_results)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
