File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant16/inputs/option_catalog.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 20
Columns: ['Family', 'Option', 'Value', 'Weight', 'BudgetUse']
Parsed column types: {'Family': 'object', 'Option': 'object', 'Value': 'int64', 'Weight': 'int64', 'BudgetUse': 'int64'}
Preview only (first 10 rows):
Family Option Value Weight BudgetUse
    G1     O1    22      6         8
    G1     O2    29      9        11
    G1     O3    31     10        13
    G1     O4    25      7         9
    G2     O1    24      7         8
    G2     O2    33     11        13
    G2     O3    28      8        10
    G2     O4    35     12        14
    G3     O1    30      9        12
    G3     O2    26      7         9
Full-file column statistics: {"Family": {"missing": 0, "unique_nonempty": 5}, "Option": {"missing": 0, "unique_nonempty": 4}, "Value": {"missing": 0, "unique_nonempty": 17, "numeric_range": [21.0, 38.0]}, "Weight": {"missing": 0, "unique_nonempty": 9, "numeric_range": [5.0, 13.0]}, "BudgetUse": {"missing": 0, "unique_nonempty": 10, "numeric_range": [7.0, 16.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "family", "matching_columns": [], "exact_matching_columns": 0}, {"term": "option", "matching_columns": [], "exact_matching_columns": 0}, {"term": "value", "matching_columns": [], "exact_matching_columns": 0}, {"term": "weight", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant16/inputs/resource_limits.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 2
Columns: ['Resource', 'Limit']
Parsed column types: {'Resource': 'object', 'Limit': 'int64'}
Preview only (first 10 rows):
 Resource Limit
   Weight    48
BudgetUse    60
Full-file column statistics: {"Resource": {"missing": 0, "unique_nonempty": 2}, "Limit": {"missing": 0, "unique_nonempty": 2, "numeric_range": [48.0, 60.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "weight", "matching_columns": [{"column": "Resource", "exact": 1, "prefix": 1, "contains": 1, "examples": ["Weight"]}], "exact_matching_columns": 1}]