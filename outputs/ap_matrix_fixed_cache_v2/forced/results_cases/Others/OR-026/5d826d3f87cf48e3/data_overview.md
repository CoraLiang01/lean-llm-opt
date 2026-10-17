File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM17/SupermarketSales.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 993
Columns: ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
Parsed column types: {'Product Name': 'object', 'Revenue': 'float64', 'Demand': 'float64', 'Initial Inventory': 'float64'}
Preview only (first 10 rows):
                Product Name Revenue Demand Initial Inventory
Electronic accessories_10.56   10.56     11                80
Electronic accessories_10.59   10.59      5                30
Electronic accessories_11.81   11.81      7                50
Electronic accessories_11.94   11.94      5                30
Electronic accessories_12.05   12.05      7                50
 Electronic accessories_12.1    12.1     11                80
Electronic accessories_12.45   12.45      8                60
Electronic accessories_13.22   13.22      7                50
Electronic accessories_13.78   13.78      5                40
Electronic accessories_14.96   14.96     12                80
Full-file column statistics: {"Product Name": {"missing": 127, "unique_nonempty": 866}, "Revenue": {"missing": 127, "unique_nonempty": 830, "numeric_range": [10.08, 99.96]}, "Demand": {"missing": 127, "unique_nonempty": 15, "numeric_range": [2.0, 17.0]}, "Initial Inventory": {"missing": 127, "unique_nonempty": 11, "numeric_range": [10.0, 120.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fashion", "matching_columns": [{"column": "Product Name", "exact": 0, "prefix": 49, "contains": 49, "examples": ["Fashion accessories_10.18", "Fashion accessories_12.09", "Fashion accessories_12.19"]}], "exact_matching_columns": 0}, {"term": "initial inventory", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revenue", "matching_columns": [], "exact_matching_columns": 0}]