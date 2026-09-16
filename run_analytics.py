"""Run e-commerce sales analytics against Parquet source data."""

import logging
import time

from src.config import settings
from src.spark_analytics import SalesAnalytics, create_spark_session


LOGGER = logging.getLogger(__name__)


def run() -> None:
    """Load Parquet sources, display all analytics, and stop Spark."""
    spark = create_spark_session()
    try:
        customers = spark.read.parquet(str(settings.raw_data_dir / "customers.parquet"))
        products = spark.read.parquet(str(settings.raw_data_dir / "products.parquet"))
        orders = spark.read.parquet(str(settings.raw_data_dir / "orders.parquet"))
        analytics = SalesAnalytics(customers, products, orders)

        operations = (
            ("top_customers", analytics.top_customers),
            ("sales_by_category", analytics.sales_by_category),
            ("monthly_trends", analytics.monthly_trends),
        )
        for name, operation in operations:
            started = time.perf_counter()
            result = operation()
            print(f"\n{name}:")
            result.show(truncate=False)
            elapsed = time.perf_counter() - started
            print(f"{name} execution time: {elapsed:.2f} seconds")
    except Exception:
        LOGGER.exception("Analytics execution failed")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()