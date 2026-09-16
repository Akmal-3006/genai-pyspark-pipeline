# E-commerce Data Pipeline

A small, reproducible e-commerce data pipeline for generating synthetic customer,
product, and order data, then analyzing it with PySpark.

## Project structure

```text
genai-pyspark-pipeline/
├── data/
│   ├── processed/          # Spark output tables
│   └── raw/                # Generated CSV input data
├── notebooks/
│   └── ecommerce_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_generator.py
│   └── spark_analytics.py
├── tests/
│   └── test_data_generator.py
├── .gitignore
└── requirements.txt
```

## Setup

Use Python 3.10 or newer, create a virtual environment, and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Java 8 or newer is required by PySpark. Verify that `java -version` works before
running the Spark job.

## Run the pipeline

Generate 1,000 customers, 100 products, and 5,000 orders:

```powershell
python -m src.data_generator --customers 1000 --products 100 --orders 5000 --seed 42
python -m src.spark_analytics
```

The generated inputs are written to `data/raw/`. Spark computes the analyzed result
tables and writes them to `data/processed/` as CSV files:

- `sales_by_category.csv`
- `monthly_sales.csv`
- `customer_summary.csv`

Run the tests with:

```powershell
pytest
```

The notebook in `notebooks/ecommerce_analysis.ipynb` provides a compact walkthrough
of the same workflow.
