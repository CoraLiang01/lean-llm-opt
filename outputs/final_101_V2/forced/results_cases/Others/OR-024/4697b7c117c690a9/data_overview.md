File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 109
Columns: ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
Parsed column types: {'Product Name': 'object', 'Revenue': 'float64', 'Demand': 'int64', 'Initial Inventory': 'int64'}
Preview only (first 10 rows):
Product Name Revenue Demand Initial Inventory
    S10_1678    95.7   1286              9440
    S10_1949   100.0   1306              9610
    S10_2016   99.91   1273              9280
    S10_4698   100.0   1262              9210
    S10_4757   100.0   1284              9520
    S10_4962   100.0   1267              9320
    S12_1099   100.0   1147              8380
    S12_1108   100.0   1326              9730
    S12_1666   100.0   1327              9720
    S12_2823   100.0   1303              9640
Full-file column statistics: {"Product Name": {"missing": 0, "unique_nonempty": 109}, "Revenue": {"missing": 0, "unique_nonempty": 57, "numeric_range": [31.2, 100.0]}, "Demand": {"missing": 0, "unique_nonempty": 98, "numeric_range": [964.0, 2403.0]}, "Initial Inventory": {"missing": 0, "unique_nonempty": 91, "numeric_range": [7140.0, 17740.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "initial inventory", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revenue", "matching_columns": [], "exact_matching_columns": 0}, {"term": "s700_", "matching_columns": [{"column": "Product Name", "exact": 0, "prefix": 12, "contains": 12, "examples": ["S700_1138", "S700_1691", "S700_1938"]}], "exact_matching_columns": 0}]