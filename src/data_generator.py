"""Generate synthetic e-commerce customers, products, and orders."""

import argparse
import logging
from datetime import date, timedelta
from typing import Sequence

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

from .config import settings


LOGGER = logging.getLogger(__name__)
CATEGORIES = ("Electronics", "Clothing", "Home", "Sports", "Books")


class SyntheticDataGenerator:
    """Generate reproducible synthetic e-commerce data as pandas DataFrames."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize Faker and NumPy random generators with a shared seed."""
        self.seed = seed
        self.faker = Faker()
        self.faker.seed_instance(seed)
        self.rng = np.random.default_rng(seed)

    def generate_customers(self, count: int = 100_000) -> pd.DataFrame:
        """Generate customers with normally distributed ages centered around 35."""
        self._validate_count(count, "customer")
        registration_start = date.today() - timedelta(days=365 * 5)
        registration_dates = [
            self.faker.date_between(start_date=registration_start, end_date="today")
            for _ in tqdm(range(count), desc="Generating registration dates")
        ]
        customers = pd.DataFrame(
            {
                "customer_id": np.arange(1, count + 1, dtype=np.int64),
                "name": [self.faker.name() for _ in tqdm(range(count), desc="Generating names")],
                "email": [self.faker.email() for _ in tqdm(range(count), desc="Generating emails")],
                "age": np.clip(self.rng.normal(35, 10, count), 18, 80).round().astype(int),
                "city": [self.faker.city() for _ in tqdm(range(count), desc="Generating cities")],
                "country": [self.faker.country() for _ in tqdm(range(count), desc="Generating countries")],
                "registration_date": registration_dates,
            }
        )
        LOGGER.info("Generated %d customers", len(customers))
        return customers

    def generate_products(self, count: int = 10_000) -> pd.DataFrame:
        """Generate products with prices from 10 to 500 and ratings from 1 to 5."""
        self._validate_count(count, "product")
        products = pd.DataFrame(
            {
                "product_id": np.arange(1, count + 1, dtype=np.int64),
                "name": [self.faker.catch_phrase() for _ in tqdm(range(count), desc="Generating products")],
                "category": self.rng.choice(CATEGORIES, size=count),
                "price": np.round(self.rng.uniform(10, 500, count), 2),
                "stock": self.rng.integers(0, 1_001, count, dtype=np.int64),
                "rating": self.rng.integers(1, 6, count, dtype=np.int64),
            }
        )
        LOGGER.info("Generated %d products", len(products))
        return products

    def generate_orders(
        self,
        customer_count: int = 100_000,
        product_count: int = 10_000,
        count: int = 1_000_000,
    ) -> pd.DataFrame:
        """Generate orders with Pareto-weighted customers following an 80/20 pattern."""
        self._validate_count(customer_count, "customer")
        self._validate_count(product_count, "product")
        self._validate_count(count, "order")

        customer_ids = np.arange(1, customer_count + 1, dtype=np.int64)
        pareto_scores = self.rng.pareto(1.16, customer_count) + 1
        top_count = max(1, int(np.ceil(customer_count * 0.20)))
        top_indices = np.argsort(pareto_scores)[-top_count:]
        probabilities = np.zeros(customer_count, dtype=np.float64)
        if customer_count == top_count:
            probabilities[:] = 1.0 / customer_count
        else:
            probabilities[:] = 0.20 / (customer_count - top_count)
            probabilities[top_indices] = 0.80 / top_count
        order_dates = pd.Timestamp.today().normalize() - pd.to_timedelta(
            self.rng.integers(0, 365, count), unit="D"
        )
        orders = pd.DataFrame(
            {
                "order_id": np.arange(1, count + 1, dtype=np.int64),
                "customer_id": self.rng.choice(customer_ids, count, p=probabilities),
                "product_id": self.rng.integers(1, product_count + 1, count, dtype=np.int64),
                "quantity": self.rng.integers(1, 11, count, dtype=np.int64),
                "order_date": order_dates,
            }
        )
        LOGGER.info("Generated %d orders using Pareto-weighted customer demand", len(orders))
        return orders

    def generate_all(
        self,
        customer_count: int = 100_000,
        product_count: int = 10_000,
        order_count: int = 1_000_000,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate and return customers, products, and orders in that order."""
        customers = self.generate_customers(customer_count)
        products = self.generate_products(product_count)
        orders = self.generate_orders(customer_count, product_count, order_count)
        return customers, products, orders

    @staticmethod
    def _validate_count(count: int, label: str) -> None:
        """Raise a clear error when a requested record count is invalid."""
        if count <= 0:
            raise ValueError(f"{label.capitalize()} count must be positive")


def generate_customers(count: int, faker: Faker) -> pd.DataFrame:
    """Compatibility wrapper for generating customers with an existing Faker instance."""
    generator = SyntheticDataGenerator()
    generator.faker = faker
    return generator.generate_customers(count)


def generate_products(count: int, faker: Faker, rng: np.random.Generator) -> pd.DataFrame:
    """Compatibility wrapper for generating products with existing random generators."""
    generator = SyntheticDataGenerator()
    generator.faker = faker
    generator.rng = rng
    return generator.generate_products(count)


def generate_orders(
    count: int,
    customer_count: int,
    product_count: int,
    faker: Faker,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Compatibility wrapper for generating orders with existing random generators."""
    generator = SyntheticDataGenerator()
    generator.faker = faker
    generator.rng = rng
    return generator.generate_orders(customer_count, product_count, count)


def generate_data(
    customer_count: int = 100_000,
    product_count: int = 10_000,
    order_count: int = 1_000_000,
    seed: int = 42,
) -> None:
    """Generate all default-scale data and save it to the configured raw directory."""
    generator = SyntheticDataGenerator(seed=seed)
    settings.ensure_directories()
    customers, products, orders = generator.generate_all(customer_count, product_count, order_count)
    customers.to_csv(settings.raw_data_dir / "customers.csv", index=False)
    products.to_csv(settings.raw_data_dir / "products.csv", index=False)
    orders.to_csv(settings.raw_data_dir / "orders.csv", index=False)
    LOGGER.info("Saved generated data to %s", settings.raw_data_dir)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the data generator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=100_000)
    parser.add_argument("--products", type=int, default=10_000)
    parser.add_argument("--orders", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    """Run the generator from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(arguments)
    generate_data(args.customers, args.products, args.orders, args.seed)


if __name__ == "__main__":
    main()
