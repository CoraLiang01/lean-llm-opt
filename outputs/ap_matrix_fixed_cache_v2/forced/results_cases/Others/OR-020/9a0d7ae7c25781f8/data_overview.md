File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM11/SalesDatainBusinesses.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 109
Columns: ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
Parsed column types: {'Product Name': 'object', 'Revenue': 'float64', 'Demand': 'int64', 'Initial Inventory': 'int64'}
Preview only (first 10 rows):
Product Name Revenue Demand Initial Inventory
    S10_1678    95.7   1280              9440
    S10_1949   100.0   1316              9610
    S10_2016   99.91   1242              9280
    S10_4698   100.0   1258              9210
    S10_4757   100.0   1292              9520
    S10_4962   100.0   1262              9320
    S12_1099   100.0   1132              8380
    S12_1108   100.0   1334              9730
    S12_1666   100.0   1323              9720
    S12_2823   100.0   1340              9640
Full-file column statistics: {"Product Name": {"missing": 0, "unique_nonempty": 109}, "Revenue": {"missing": 0, "unique_nonempty": 57, "numeric_range": [31.2, 100.0]}, "Demand": {"missing": 0, "unique_nonempty": 96, "numeric_range": [957.0, 2457.0]}, "Initial Inventory": {"missing": 0, "unique_nonempty": 91, "numeric_range": [7140.0, 17740.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "initial inventory", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revenue", "matching_columns": [], "exact_matching_columns": 0}]