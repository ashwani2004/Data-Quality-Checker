# Data Observability Hub

A beginner-friendly web application built with `Streamlit` to upload datasets, detect data quality issues, clean the data, visualize patterns, and download the cleaned result.

## Features

- Upload CSV and Excel files
- Preview the dataset in an interactive table
- View dataset shape and column data types
- Check missing values, duplicates, outliers, and type mismatches
- Detect class imbalance for an optional target column
- See visualizations such as missing value heatmap, distributions, and correlation heatmap
- Apply simple cleaning actions
- Download the cleaned dataset as CSV

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Project Files

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies

## Run Locally

1. Create and activate a virtual environment if you want:

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the Streamlit app:

```bash
streamlit run app.py
```

4. Open the local URL shown in the terminal, usually:


## Notes

- Excel upload support uses `openpyxl`.
- The app stores a working copy of the uploaded data in the Streamlit session.
- You can reset the cleaned data back to the original uploaded file from the sidebar.
