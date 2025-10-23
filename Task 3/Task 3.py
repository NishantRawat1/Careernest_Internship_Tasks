"""
Advanced Supermarket Sales Analysis
File: advanced_supermarket_analysis.py
Author: Nishant Rawat | #InternWithCareernest

Usage:
    - Place "SuperMarket Analysis.csv" in the same folder.
    - Install requirements: pip install -r requirements.txt
    - Run: python advanced_supermarket_analysis.py
Outputs:
    - ./outputs/ : saved charts (.png)
    - cleaned_supermarket_sales.csv : cleaned dataset
    - analysis_report.md : short markdown report with key findings
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---- Config ----
INPUT_CSV = "SuperMarket Analysis.csv"
CLEANED_CSV = "cleaned_supermarket_sales.csv"
OUT_DIR = "outputs"
REPORT_MD = "analysis_report.md"
RANDOM_STATE = 42

os.makedirs(OUT_DIR, exist_ok=True)
sns.set(style="whitegrid")

def safe_print(msg):
    print(msg)

# ---- Load dataset ----
safe_print("Loading dataset...")
df = pd.read_csv(INPUT_CSV)
safe_print(f"Initial shape: {df.shape}")

# ---- Normalize column names (strip and lower) ----
df.columns = [c.strip() for c in df.columns]

# --- Quick look ---
safe_print("\nColumns:")
safe_print(df.columns.tolist())
safe_print("\nFirst 5 rows:")
safe_print(df.head().to_string(index=False))

# ---- Basic info ----
safe_print("\nData types & non-null counts:")
safe_print(df.info(verbose=False))

# ---- Standardize expected column names (attempt) ----
# We'll try to handle common variations like 'Product line' vs 'Product Line', 'Unit price' vs 'Unit Price'
col_map = {}
for c in df.columns:
    lc = c.lower().replace(" ", "")
    if "invoice" in lc:
        col_map[c] = "InvoiceID"
    if "branch" in lc:
        col_map[c] = "Branch"
    if "city" in lc:
        col_map[c] = "City"
    if "customertype" in lc or "customer type" in c.lower():
        col_map[c] = "CustomerType"
    if "gender" in lc:
        col_map[c] = "Gender"
    if "product" in lc and "line" in lc:
        col_map[c] = "ProductLine"
    if "unitprice" in lc or "unit price" in c.lower():
        col_map[c] = "UnitPrice"
    if "quantity" in lc:
        col_map[c] = "Quantity"
    if "tax" in lc:
        col_map[c] = "Tax"
    if "total" == lc or "total" in lc:
        col_map[c] = "Total"
    if "date" in lc and "time" not in lc:
        col_map[c] = "Date"
    if "time" in lc:
        col_map[c] = "Time"
    if "payment" in lc:
        col_map[c] = "Payment"
    if "rating" in lc:
        col_map[c] = "Rating"
    if "cogs" in lc:
        col_map[c] = "COGS"
    if "gross" in lc and "margin" in lc:
        col_map[c] = "GrossMarginPercentage"
    if "grossincome" in lc or "gross income" in c.lower():
        col_map[c] = "GrossIncome"

# Apply rename for columns found
df.rename(columns=col_map, inplace=True)
safe_print("\nRenamed columns (applied):")
safe_print(col_map)

# ---- Data cleaning steps ----
safe_print("\n--- Data Cleaning ---")

# 1) Drop exact duplicates
dups = df.duplicated().sum()
safe_print(f"Duplicate rows found: {dups}")
if dups > 0:
    df.drop_duplicates(inplace=True)

# 2) Strip whitespace from string columns
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].astype(str).str.strip()

# 3) Parse Date column if available
if "Date" in df.columns:
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month_name()
        df['Day'] = df['Date'].dt.day
        df['Weekday'] = df['Date'].dt.day_name()
    except Exception as e:
        safe_print("Could not parse Date column automatically. Error:", e)

# 4) Ensure numeric columns are numeric
num_cols = []
for possible in ["UnitPrice", "Unit Price", "Unit_Price", "Quantity", "Total", "Rating", "COGS", "GrossIncome"]:
    if possible in df.columns:
        num_cols.append(possible)
# Also attempt to coerce common variations
for c in df.columns:
    if c not in num_cols:
        try:
            # if values look numeric in >75% rows, coerce
            sample = df[c].dropna().astype(str).head(50).str.replace(',', '').str.replace('$', '')
            numeric_frac = sample.str.replace('.', '', 1).str.isnumeric().mean()
            if numeric_frac > 0.75:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '').str.replace('$', ''), errors='coerce')
                num_cols.append(c)
        except Exception:
            pass

safe_print(f"Numeric-like columns detected/coerced: {num_cols}")

# 5) Fill or drop missing values sensibly
missing_before = df.isnull().sum()
safe_print("\nMissing values before handling:\n" + missing_before.to_string())
# Strategy:
# - If 'Total' missing but UnitPrice & Quantity present, compute Total
if 'Total' in df.columns and df['Total'].isnull().any():
    if 'UnitPrice' in df.columns and 'Quantity' in df.columns:
        mask = df['Total'].isnull() & df['UnitPrice'].notnull() & df['Quantity'].notnull()
        df.loc[mask, 'Total'] = df.loc[mask, 'UnitPrice'] * df.loc[mask, 'Quantity']
        safe_print(f"Filled {mask.sum()} Total values using UnitPrice * Quantity")

# - For numeric columns, fill small count of missing with median
for c in df.select_dtypes(include=[np.number]).columns:
    if df[c].isnull().sum() > 0 and df[c].isnull().sum() < len(df)*0.1:
        median = df[c].median()
        df[c].fillna(median, inplace=True)
        safe_print(f"Filled {df[c].isnull().sum()} NA in {c} with median {median:.2f}")

# - For categorical columns with few missing, fill with 'Unknown'
for c in df.select_dtypes(include="object").columns:
    if df[c].isnull().sum() > 0 and df[c].isnull().sum() < len(df)*0.1:
        df[c].fillna("Unknown", inplace=True)

missing_after = df.isnull().sum()
safe_print("\nMissing values after handling:\n" + missing_after.to_string())

# ---- Feature engineering ----
safe_print("\n--- Feature Engineering ---")
# Ensure Total exists
if 'Total' not in df.columns and 'UnitPrice' in df.columns and 'Quantity' in df.columns:
    df['Total'] = df['UnitPrice'] * df['Quantity']
    safe_print("Created Total = UnitPrice * Quantity")

# Create a simple "HighSpender" flag: Total > 75th percentile
if 'Total' in df.columns:
    thr = df['Total'].quantile(0.75)
    df['HighSpender'] = (df['Total'] > thr).astype(int)
    safe_print(f"HighSpender threshold (75th pct): {thr:.2f}")

# RFM-like features (if InvoiceID present)
if 'InvoiceID' in df.columns and 'Total' in df.columns and 'Date' in df.columns:
    snapshot_date = df['Date'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('InvoiceID').agg({
        'Date': lambda x: (snapshot_date - x.max()).days,
        'InvoiceID': 'count',
        'Total': 'sum'
    }).rename(columns={'Date':'Recency','InvoiceID':'Frequency','Total':'Monetary'})
    # join RFM back on InvoiceID - optional
    safe_print("RFM sample (first 5):")
    safe_print(rfm.head().to_string())

# ---- Exploratory Data Analysis (plots saved to ./outputs) ----
safe_print("\n--- Exploratory Data Analysis & Plots ---")

# Utility to save figures
def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    safe_print(f"Saved: {path}")

# 1) Total sales by Branch (bar)
if 'Branch' in df.columns and 'Total' in df.columns:
    fig, ax = plt.subplots(figsize=(7,4))
    order = df.groupby('Branch')['Total'].sum().sort_values(ascending=False).index
    sns.barplot(x='Branch', y='Total', data=df, estimator=sum, order=order, ax=ax)
    ax.set_title("Total Sales by Branch")
    ax.set_ylabel("Total Sales")
    save_fig(fig, "total_sales_by_branch.png")
    plt.close(fig)

# 2) Sales by City
if 'City' in df.columns and 'Total' in df.columns:
    fig, ax = plt.subplots(figsize=(8,4))
    order = df.groupby('City')['Total'].sum().sort_values(ascending=False).index
    sns.barplot(x='City', y='Total', data=df, estimator=sum, order=order, ax=ax)
    ax.set_title("Total Sales by City")
    ax.set_ylabel("Total Sales")
    save_fig(fig, "total_sales_by_city.png")
    plt.close(fig)

# 3) Sales by Product Line (top 10)
if 'ProductLine' in df.columns and 'Total' in df.columns:
    prod_sales = df.groupby('ProductLine')['Total'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x=prod_sales.values, y=prod_sales.index, ax=ax)
    ax.set_title("Sales by Product Line (descending)")
    ax.set_xlabel("Total Sales")
    save_fig(fig, "sales_by_productline.png")
    plt.close(fig)

# 4) Payment method distribution
if 'Payment' in df.columns:
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x='Payment', data=df, order=df['Payment'].value_counts().index, ax=ax)
    ax.set_title("Payment Method Counts")
    save_fig(fig, "payment_method_counts.png")
    plt.close(fig)

# 5) Sales by Gender
if 'Gender' in df.columns and 'Total' in df.columns:
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x='Gender', y='Total', data=df, estimator=sum, order=df['Gender'].unique(), ax=ax)
    ax.set_title("Total Sales by Gender")
    save_fig(fig, "sales_by_gender.png")
    plt.close(fig)

# 6) Monthly sales trend (if Month exists)
if 'Month' in df.columns and 'Total' in df.columns:
    month_order = ["January","February","March","April","May","June","July","August","September","October","November","December"]
    monthly = df.groupby('Month')['Total'].sum().reindex([m for m in month_order if m in df['Month'].unique()])
    fig, ax = plt.subplots(figsize=(10,4))
    sns.lineplot(x=monthly.index, y=monthly.values, marker='o', ax=ax)
    ax.set_title("Monthly Sales Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")
    save_fig(fig, "monthly_sales_trend.png")
    plt.close(fig)

# 7) Rating distribution & relation with Total
if 'Rating' in df.columns:
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df['Rating'].dropna(), kde=True, ax=ax)
    ax.set_title("Customer Rating Distribution")
    save_fig(fig, "rating_distribution.png")
    plt.close(fig)

    if 'Total' in df.columns:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(x='Rating', y='Total', data=df.sample(min(500, len(df))), ax=ax)
        ax.set_title("Rating vs Total (sample)")
        save_fig(fig, "rating_vs_total.png")
        plt.close(fig)

# 8) Correlation heatmap (numeric)
fig, ax = plt.subplots(figsize=(8,6))
num_df = df.select_dtypes(include=[np.number])
if not num_df.empty:
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title("Numeric Features Correlation")
    save_fig(fig, "correlation_heatmap.png")
    plt.close(fig)

# ---- Simple segmentation: KMeans on numerical spending features ----
safe_print("\n--- Simple Customer Segmentation (KMeans) ---")
seg_df = df.copy()
# choose features: Total, Quantity, Rating (if exist)
features = []
for f in ['Total','Quantity','Rating','GrossIncome']:
    if f in seg_df.columns:
        features.append(f)
safe_print(f"Segmentation features used: {features}")

if len(features) >= 2:
    X = seg_df[features].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE)
    seg_labels = kmeans.fit_predict(Xs)
    seg_df['Segment'] = seg_labels
    # save small summary
    seg_summary = seg_df.groupby('Segment')[features].agg(['mean','median','count'])
    safe_print("\nSegment summary:")
    safe_print(seg_summary.to_string())
    # save segment plot (Total vs Quantity)
    if 'Total' in seg_df.columns and 'Quantity' in seg_df.columns:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(x='Quantity', y='Total', hue='Segment', data=seg_df.sample(min(1000, len(seg_df))), palette='Set2', ax=ax)
        ax.set_title("KMeans Segments (Quantity vs Total) - sample")
        save_fig(fig, "kmeans_segments_qty_total.png")
        plt.close(fig)
else:
    safe_print("Not enough numeric features available for segmentation.")

# ---- Save cleaned data ----
safe_print("\nSaving cleaned dataset...")
df.to_csv(CLEANED_CSV, index=False)
safe_print(f"Saved cleaned CSV -> {CLEANED_CSV}")

# ---- Generate a short markdown report ----
safe_print("\nGenerating markdown report...")
with open(REPORT_MD, "w", encoding="utf-8") as f:
    f.write("# Supermarket Sales — Analysis Report\n\n")
    f.write("**Author:** Nishant Rawat  \n")
    f.write("**Internship:** Careernest Pvt. Ltd.  \n\n")
    f.write("## 1) Dataset overview\n")
    f.write(f"- Rows: {df.shape[0]}  \n- Columns: {df.shape[1]}  \n\n")
    f.write("## 2) Key data cleaning steps\n")
    f.write("- Removed duplicates (if any).  \n")
    f.write("- Standardized/renamed common columns (Branch, City, ProductLine, UnitPrice, Quantity, Total, Rating).  \n")
    f.write("- Parsed Date and created Month/Weekday features.  \n")
    f.write("- Filled some missing numeric values with median and computed Total where possible.  \n\n")

    f.write("## 3) Exploratory findings & charts\n")
    if 'Branch' in df.columns and 'Total' in df.columns:
        top_branch = df.groupby('Branch')['Total'].sum().idxmax()
        f.write(f"- **Top branch by revenue:** {top_branch}. See `outputs/total_sales_by_branch.png`.  \n")
    if 'City' in df.columns and 'Total' in df.columns:
        top_city = df.groupby('City')['Total'].sum().idxmax()
        f.write(f"- **Top city by revenue:** {top_city}. See `outputs/total_sales_by_city.png`.  \n")
    if 'ProductLine' in df.columns:
        top_product = df.groupby('ProductLine')['Total'].sum().sort_values(ascending=False).head(1).index[0]
        f.write(f"- **Top product line:** {top_product}. See `outputs/sales_by_productline.png`.  \n")
    if 'Payment' in df.columns:
        top_payment = df['Payment'].value_counts().idxmax()
        f.write(f"- **Most common payment method:** {top_payment}. See `outputs/payment_method_counts.png`.  \n")
    if 'Gender' in df.columns:
        gender_sales = df.groupby('Gender')['Total'].sum().sort_values(ascending=False)
        f.write(f"- **Sales by gender:** {gender_sales.to_dict()} (See `outputs/sales_by_gender.png`).  \n")
    if 'Rating' in df.columns:
        f.write("- **Ratings:** distribution saved in `outputs/rating_distribution.png`.  \n")
    f.write("\n## 4) Segmentation\n")
    if 'Segment' in df.columns or 'Segment' in seg_df.columns:
        f.write("- Performed KMeans segmentation (k=3) using numeric features (Total, Quantity, Rating). See `outputs/kmeans_segments_qty_total.png`.\n\n")
        f.write("## 5) Business insights & recommendations\n")
        f.write("- Focus promotions on the top branch & product lines to increase revenue.  \n")
        f.write("- Encourage payment methods that bring higher average order values.  \n")
        f.write("- Tailor marketing for 'HighSpender' customers (75th percentile) with loyalty offers.  \n")
        f.write("- Investigate why some branches or product lines have low ratings and address service issues.  \n")
    else:
        f.write("- Segmentation was not possible due to insufficient numeric features.\n\n")

    f.write("## 6) Files & outputs\n")
    f.write("- Cleaned CSV: `cleaned_supermarket_sales.csv`  \n")
    f.write("- Plots: saved in `outputs/`  \n")
    f.write("- Full code: `advanced_supermarket_analysis.py`  \n\n")

safe_print(f"Saved short report -> {REPORT_MD}")

safe_print("\nAll done ✅")
safe_print(f"Check the `{OUT_DIR}` folder for generated chart images, `{CLEANED_CSV}` for cleaned data, and `{REPORT_MD}` for the report.")
