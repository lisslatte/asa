# Customer Segmentation Streamlit App

This app lets a user upload 4 CSV files:

- `baskets`
- `category spends`
- `customers`
- `lineitems`

It then:

- validates the uploaded files
- runs the segmentation pipeline
- predicts a segment for every customer
- shows charts and summary tables
- exports `customer_number` and `segment` as a CSV

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run streamlit_app.py
```

## Output

The downloadable output file is:

- `customer_segments_output.csv`

with columns:

- `customer_number`
- `segment`
