File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 161
Columns: ['Product_Reference', 'Revenue', 'Demand', 'Initial Inventory']
Parsed column types: {'Product_Reference': 'object', 'Revenue': 'float64', 'Demand': 'int64', 'Initial Inventory': 'float64'}
Preview only (first 10 rows):
Product_Reference Revenue Demand Initial Inventory
 ELE-ACC-10000478     2.0    140            1000.0
 ELE-ACC-10002424     2.0    136            1000.0
 ELE-ACC-10018567     2.0    268            2000.0
 ELE-ACC-10019567     2.0    121            1000.0
 ELE-CAM-10000475    12.0    812            6000.0
 ELE-CAM-10002121    12.0    875            6000.0
 ELE-CAM-10015234    12.0   1519           12000.0
 ELE-CAM-10016234    12.0    868            6000.0
 ELE-HEA-10000460     3.0    192            1500.0
 ELE-HEA-10000493     3.6    230            1800.0
Full-file column statistics: {"Product_Reference": {"missing": 0, "unique_nonempty": 161}, "Revenue": {"missing": 0, "unique_nonempty": 27, "numeric_range": [0.15, 18.0]}, "Demand": {"missing": 0, "unique_nonempty": 141, "numeric_range": [7.0, 3965.0]}, "Initial Inventory": {"missing": 0, "unique_nonempty": 45, "numeric_range": [50.0, 30000.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "ele-s", "matching_columns": [{"column": "Product_Reference", "exact": 0, "prefix": 12, "contains": 12, "examples": ["ELE-SMA-10000463", "ELE-SMA-10000487", "ELE-SMA-10003333"]}], "exact_matching_columns": 0}, {"term": "initial inventory", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revenue", "matching_columns": [], "exact_matching_columns": 0}]