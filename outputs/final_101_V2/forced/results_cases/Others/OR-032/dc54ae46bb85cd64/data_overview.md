File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 40
Columns: ['Product_Name', 'Revenue', 'Demand', 'Initial Inventory']
Parsed column types: {'Product_Name': 'object', 'Revenue': 'float64', 'Demand': 'int64', 'Initial Inventory': 'float64'}
Preview only (first 10 rows):
    Product_Name Revenue Demand Initial Inventory
     Books_15.15   15.15   1980            9920.0
      Books_30.3    30.3   3024           20160.0
     Books_45.45   45.45   4536           30000.0
      Books_60.6    60.6   5601           38360.0
     Books_75.75   75.75   7567           51450.0
Clothing_1200.32 1200.32  39904          273960.0
 Clothing_1500.4  1500.4  50929          347000.0
 Clothing_300.08  300.08  13719           68690.0
 Clothing_600.16  600.16  20857          139050.0
 Clothing_900.24  900.24  31483          207280.0
Full-file column statistics: {"Product_Name": {"missing": 0, "unique_nonempty": 40}, "Revenue": {"missing": 0, "unique_nonempty": 40, "numeric_range": [5.23, 5250.0]}, "Demand": {"missing": 0, "unique_nonempty": 40, "numeric_range": [1970.0, 50929.0]}, "Initial Inventory": {"missing": 0, "unique_nonempty": 39, "numeric_range": [9850.0, 347000.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "books", "matching_columns": [{"column": "Product_Name", "exact": 0, "prefix": 5, "contains": 5, "examples": ["Books_15.15", "Books_30.3", "Books_45.45"]}], "exact_matching_columns": 0}, {"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "initial inventory", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revenue", "matching_columns": [], "exact_matching_columns": 0}]