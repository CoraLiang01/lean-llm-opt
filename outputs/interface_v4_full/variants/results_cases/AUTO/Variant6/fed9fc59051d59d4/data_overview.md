File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant6/inputs/option_catalog.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 24
Columns: ['Family', 'Option', 'Value', 'Weight', 'BudgetUse']
Parsed column types: {'Family': 'object', 'Option': 'object', 'Value': 'int64', 'Weight': 'int64', 'BudgetUse': 'int64'}
Preview only (first 10 rows):
Family Option Value Weight BudgetUse
    F1     O1    28      8        10
    F1     O2    34     11        13
    F1     O3    30      9        12
    F1     O4    24      7         9
    F2     O1    25      7         9
    F2     O2    31     10        12
    F2     O3    36     12        14
    F2     O4    29      9        11
    F3     O1    33     10        12
    F3     O2    27      8        10
Full-file column statistics: {"Family": {"missing": 0, "unique_nonempty": 6}, "Option": {"missing": 0, "unique_nonempty": 4}, "Value": {"missing": 0, "unique_nonempty": 15, "numeric_range": [24.0, 38.0]}, "Weight": {"missing": 0, "unique_nonempty": 8, "numeric_range": [6.0, 13.0]}, "BudgetUse": {"missing": 0, "unique_nonempty": 8, "numeric_range": [8.0, 15.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "family", "matching_columns": [], "exact_matching_columns": 0}, {"term": "option", "matching_columns": [], "exact_matching_columns": 0}, {"term": "value", "matching_columns": [], "exact_matching_columns": 0}, {"term": "weight", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant6/inputs/resource_limits.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 2
Columns: ['Resource', 'Limit']
Parsed column types: {'Resource': 'object', 'Limit': 'int64'}
Preview only (first 10 rows):
 Resource Limit
   Weight    55
BudgetUse    70
Full-file column statistics: {"Resource": {"missing": 0, "unique_nonempty": 2}, "Limit": {"missing": 0, "unique_nonempty": 2, "numeric_range": [55.0, 70.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "weight", "matching_columns": [{"column": "Resource", "exact": 1, "prefix": 1, "contains": 1, "examples": ["Weight"]}], "exact_matching_columns": 1}]