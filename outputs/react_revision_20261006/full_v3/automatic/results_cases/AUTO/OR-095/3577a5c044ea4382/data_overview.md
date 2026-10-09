File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 3
Columns: ['Resource', 'MonthlyLimit']
Parsed column types: {'Resource': 'object', 'MonthlyLimit': 'int64'}
Preview only (first 10 rows):
  Resource MonthlyLimit
LaborHours         5000
 MaterialA        24000
 MaterialB        15000
Full-file column statistics: {"Resource": {"missing": 0, "unique_nonempty": 3}, "MonthlyLimit": {"missing": 0, "unique_nonempty": 3, "numeric_range": [5000.0, 24000.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: []

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 141
Columns: ['Product', 'LaborHours', 'MaterialA', 'MaterialB', 'Profit']
Parsed column types: {'Product': 'object', 'LaborHours': 'float64', 'MaterialA': 'int64', 'MaterialB': 'int64', 'Profit': 'int64'}
Preview only (first 10 rows):
 Product LaborHours MaterialA MaterialB Profit
 Widget1        1.6        24        14    525
 Widget2          2        20        10    678
 Widget3        2.5        12        18    812
 Widget4       1.9         21        15    769
 Widget5       0.0         15        26    952
 Widget6       0.1         24        17    987
 Widget7       1.2         15        30    644
 Widget8       1.3         21        24    795
 Widget9       0.4         20        30    829
Widget10       0.9         18        27    574
Full-file column statistics: {"Product": {"missing": 0, "unique_nonempty": 141}, "LaborHours": {"missing": 0, "unique_nonempty": 24, "numeric_range": [0.0, 2.5]}, "MaterialA": {"missing": 0, "unique_nonempty": 16, "numeric_range": [10.0, 25.0]}, "MaterialB": {"missing": 0, "unique_nonempty": 18, "numeric_range": [10.0, 30.0]}, "Profit": {"missing": 0, "unique_nonempty": 119, "numeric_range": [504.0, 1000.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "profit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "widget1", "matching_columns": [{"column": "Product", "exact": 1, "prefix": 53, "contains": 53, "examples": ["Widget1", "Widget10", "Widget11"]}], "exact_matching_columns": 1}, {"term": "widget141", "matching_columns": [{"column": "Product", "exact": 1, "prefix": 1, "contains": 1, "examples": ["Widget141"]}], "exact_matching_columns": 1}, {"term": "widget3", "matching_columns": [{"column": "Product", "exact": 1, "prefix": 11, "contains": 11, "examples": ["Widget3", "Widget30", "Widget31"]}], "exact_matching_columns": 1}]