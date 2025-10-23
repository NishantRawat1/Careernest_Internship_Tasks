# Supermarket Sales — Analysis Report

**Author:** Nishant Rawat  
**Internship:** Careernest Pvt. Ltd.  

## 1) Dataset overview
- Rows: 1000  
- Columns: 22  

## 2) Key data cleaning steps
- Removed duplicates (if any).  
- Standardized/renamed common columns (Branch, City, ProductLine, UnitPrice, Quantity, Total, Rating).  
- Parsed Date and created Month/Weekday features.  
- Filled some missing numeric values with median and computed Total where possible.  

## 3) Exploratory findings & charts
- **Top branch by revenue:** Giza. See `outputs/total_sales_by_branch.png`.  
- **Top city by revenue:** Naypyitaw. See `outputs/total_sales_by_city.png`.  
- **Top product line:** Food and beverages. See `outputs/sales_by_productline.png`.  
- **Most common payment method:** Ewallet. See `outputs/payment_method_counts.png`.  
- **Sales by gender:** {'Female': 185401.75, 'Male': 122185.63} (See `outputs/sales_by_gender.png`).  
- **Ratings:** distribution saved in `outputs/rating_distribution.png`.  

## 4) Segmentation
- Performed KMeans segmentation (k=3) using numeric features (Total, Quantity, Rating). See `outputs/kmeans_segments_qty_total.png`.

## 5) Business insights & recommendations
- Focus promotions on the top branch & product lines to increase revenue.  
- Encourage payment methods that bring higher average order values.  
- Tailor marketing for 'HighSpender' customers (75th percentile) with loyalty offers.  
- Investigate why some branches or product lines have low ratings and address service issues.  
## 6) Files & outputs
- Cleaned CSV: `cleaned_supermarket_sales.csv`  
- Plots: saved in `outputs/`  
- Full code: `advanced_supermarket_analysis.py`  

