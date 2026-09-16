"""Tests for synthetic data generation."""

import pandas as pd
import pytest
from faker import Faker
import numpy as np

from src.config import Settings
from src.data_generator import generate_customers, generate_data, generate_orders, generate_products


def test_generated_tables_have_expected_relationships() -> None:
    """Generated order references should point to existing entity IDs."""
    faker = Faker()
    rng = np.random.default_rng(7)
    customers = generate_customers(4, faker)
    products = generate_products(3, faker, rng)
    orders = generate_orders(10, len(customers), len(products), faker, rng)

    assert set(orders["customer_id"]).issubset(set(customers["customer_id"]))
    assert set(orders["product_id"]).issubset(set(products["product_id"]))
    assert (orders["quantity"] > 0).all()


def test_generate_data_writes_csv_files(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The public generator should write all three raw input files."""
    from src import data_generator

    monkeypatch.setattr(data_generator, "settings", Settings(project_root=tmp_path))
    generate_data(customer_count=2, product_count=2, order_count=3, seed=1)

    raw_data_dir = tmp_path / "data" / "raw"
    for filename in ("customers.csv", "products.csv", "orders.csv"):
        assert (raw_data_dir / filename).exists()
        assert not pd.read_csv(raw_data_dir / filename).empty