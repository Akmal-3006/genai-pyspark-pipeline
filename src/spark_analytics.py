"""Analyze generated e-commerce data with PySpark."""

import argparse
import csv
import logging
from pathlib import Path
from typing import Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, DoubleType, IntegerType, StringType, StructField, StructType

from .config import settings


LOGGER = logging.getLogger(__name__)


class SalesAnalytics:
    """Run reusable sales analyses over generated e-commerce DataFrames."""

    def __init__(
        self,
        customers: DataFrame,
        products: DataFrame,
        orders: DataFrame,
    ) -> None:
        """Initialize analytics with customer, product, and order DataFrames."""
        self.customers = customers
        self.products = products
        self.orders = orders

    def _order_facts(self) -> DataFrame:
        """Join orders to products and calculate non-cancelled order revenue."""
        order_columns = set(self.orders.columns)
        active_orders = self.orders
        if "status" in order_columns:
            active_orders = active_orders.filter(F.col("status") != "cancelled")
        return (
            active_orders.join(self.products, on="product_id", how="inner")
            .withColumn("revenue", F.round(F.col("quantity") * F.col("price"), 2))
        )

    def top_customers(self, limit: int = 10) -> DataFrame:
        """Return customers ranked by lifetime revenue."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        facts = self._order_facts()
        return (
            facts.groupBy("customer_id")
            .agg(
                F.countDistinct("order_id").alias("order_count"),
                F.round(F.sum("revenue"), 2).alias("lifetime_value"),
            )
            .join(self.customers.select("customer_id", "name", "email"), "customer_id")
            .select("customer_id", "name", "email", "order_count", "lifetime_value")
            .orderBy(F.desc("lifetime_value"))
            .limit(limit)
        )

    def sales_by_category(self) -> DataFrame:
        """Return units sold and revenue grouped by product category."""
        return (
            self._order_facts()
            .groupBy("category")
            .agg(
                F.sum("quantity").alias("units_sold"),
                F.round(F.sum("revenue"), 2).alias("revenue"),
            )
            .orderBy(F.desc("revenue"))
        )

    def monthly_trends(self) -> DataFrame:
        """Return order count and revenue grouped by calendar month."""
        return (
            self._order_facts()
            .withColumn("month", F.date_format("order_date", "yyyy-MM"))
            .groupBy("month")
            .agg(
                F.countDistinct("order_id").alias("orders"),
                F.round(F.sum("revenue"), 2).alias("revenue"),
            )
            .orderBy("month")
        )


def create_spark_session() -> SparkSession:
    """Create a Spark session tuned for a 16 GB RAM, 8-core laptop.

    The driver receives 8 GB, leaving memory for Windows and other applications.
    Thirty-two initial shuffle partitions provide about four tasks per core for
    groupBy operations; adaptive execution can coalesce them at runtime.
    """
    return (
        SparkSession.builder
        .appName("EcommerceAnalytics")
        .master("local[8]")
        # Reserve roughly half of system RAM for Spark while leaving headroom for the OS.
        .config("spark.driver.memory", "8g")
        # Start with 4 partitions per core for groupBy and join shuffles.
        .config("spark.sql.shuffle.partitions", "32")
        # Let Spark optimize query plans and partition sizes at runtime.
        .config("spark.sql.adaptive.enabled", "true")
        # Kryo is faster and more compact than Java serialization for Spark data.
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        # Merge small shuffle partitions after adaptive statistics are available.
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )


def read_source_data(spark: SparkSession) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Read customers, products, and orders from the raw CSV directory."""
    customer_schema = StructType([
        StructField("customer_id", IntegerType(), False),
        StructField("name", StringType(), True),
        StructField("email", StringType(), True),
        StructField("city", StringType(), True),
        StructField("country", StringType(), True),
        StructField("signup_date", DateType(), True),
    ])
    product_schema = StructType([
        StructField("product_id", IntegerType(), False),
        StructField("product_name", StringType(), True),
        StructField("category", StringType(), True),
        StructField("price", DoubleType(), True),
    ])
    order_schema = StructType([
        StructField("order_id", IntegerType(), False),
        StructField("customer_id", IntegerType(), False),
        StructField("product_id", IntegerType(), False),
        StructField("quantity", IntegerType(), False),
        StructField("order_date", DateType(), True),
        StructField("status", StringType(), True),
    ])
    reader = spark.read.option("header", True)
    return (
        reader.schema(customer_schema).csv(str(settings.raw_data_dir / "customers.csv")),
        reader.schema(product_schema).csv(str(settings.raw_data_dir / "products.csv")),
        reader.schema(order_schema).csv(str(settings.raw_data_dir / "orders.csv")),
    )


def build_order_facts(orders: DataFrame, products: DataFrame) -> DataFrame:
    """Join orders to products and calculate revenue for each order line."""
    return (
        orders.join(products, on="product_id", how="inner")
        .withColumn("revenue", F.round(F.col("quantity") * F.col("price"), 2))
        .filter(F.col("status") != "cancelled")
    )


def build_insights(orders: DataFrame, products: DataFrame) -> dict[str, DataFrame]:
    """Build business insight tables from the source DataFrames."""
    facts = build_order_facts(orders, products)
    return {
        "sales_by_category": facts.groupBy("category").agg(
            F.sum("quantity").alias("units_sold"),
            F.round(F.sum("revenue"), 2).alias("revenue"),
        ).orderBy(F.desc("revenue")),
        "monthly_sales": facts.withColumn("month", F.date_format("order_date", "yyyy-MM")).groupBy("month").agg(
            F.countDistinct("order_id").alias("orders"),
            F.round(F.sum("revenue"), 2).alias("revenue"),
        ).orderBy("month"),
        "customer_summary": facts.groupBy("customer_id").agg(
            F.countDistinct("order_id").alias("order_count"),
            F.round(F.sum("revenue"), 2).alias("lifetime_value"),
        ).orderBy(F.desc("lifetime_value")),
    }


def write_result(dataframe: DataFrame, output_path: Path) -> None:
    """Write a Spark result as a single CSV file without Hadoop filesystem APIs."""
    rows = [row.asDict() for row in dataframe.collect()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=dataframe.columns)
        writer.writeheader()
        writer.writerows(rows)


def run_analytics() -> None:
    """Read raw data, write insight tables, and stop the Spark session."""
    settings.ensure_directories()
    spark = create_spark_session()
    try:
        _, products, orders = read_source_data(spark)
        for name, dataframe in build_insights(orders, products).items():
            output_path = settings.processed_data_dir / f"{name}.csv"
            write_result(dataframe, output_path)
            LOGGER.info("Wrote %s", output_path)
    finally:
        spark.stop()


def main(arguments: Sequence[str] | None = None) -> None:
    """Run the Spark analytics job from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(arguments)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_analytics()


if __name__ == "__main__":
    main()