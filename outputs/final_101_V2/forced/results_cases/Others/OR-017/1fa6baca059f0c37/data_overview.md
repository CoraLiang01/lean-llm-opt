File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv
CSV delimiter: comma; first line consumed as header.
Total rows: 5242
Columns: ['SKU', 'Revenue', 'Demand', 'Initial Inventory']
Parsed column types: {'SKU': 'object', 'Revenue': 'float64', 'Demand': 'int64', 'Initial Inventory': 'float64'}
Preview only (first 10 rows):
  SKU Revenue Demand Initial Inventory
00GVC   17.68      4              20.0
00OK1    1.27     33             180.0
0121I    2.03     60             310.0
01IEO    4.96     81             430.0
01IQT     1.5     14              70.0
01L05  167.64     16             100.0
01V7M     8.0     88             450.0
01XVY    1.46      2              10.0
029WA    5.03      4              20.0
03C6L    1.23     68             360.0
Full-file column statistics: {"SKU": {"missing": 0, "unique_nonempty": 5242}, "Revenue": {"missing": 0, "unique_nonempty": 1942, "numeric_range": [0.02, 531.27]}, "Demand": {"missing": 0, "unique_nonempty": 458, "numeric_range": [1.0, 7844.0]}, "Initial Inventory": {"missing": 0, "unique_nonempty": 333, "numeric_range": [10.0, 57700.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "initial inventory", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revenue", "matching_columns": [], "exact_matching_columns": 0}, {"term": "zz", "matching_columns": [{"column": "SKU", "exact": 0, "prefix": 5, "contains": 18, "examples": ["2L6ZZ", "2ZZWJ", "6BZZ3"]}], "exact_matching_columns": 0}]