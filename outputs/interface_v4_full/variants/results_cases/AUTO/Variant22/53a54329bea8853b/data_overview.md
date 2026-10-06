File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant22/inputs/option_catalog.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 18
Columns: ['Family', 'Option', 'Value', 'Weight', 'LaborHours']
Parsed column types: {'Family': 'object', 'Option': 'object', 'Value': 'int64', 'Weight': 'int64', 'LaborHours': 'int64'}
Preview only (first 10 rows):
Family Option Value Weight LaborHours
    C1     O1    18      5          6
    C1     O2    27      8          9
    C1     O3    30     10         11
    C2     O1    24      7          8
    C2     O2    32     11         13
    C2     O3    28      9         10
    C3     O1    22      6          7
    C3     O2    35     12         14
    C3     O3    31     10         12
    C4     O1    20      5          8
Full-file column statistics: {"Family": {"missing": 0, "unique_nonempty": 6}, "Option": {"missing": 0, "unique_nonempty": 3}, "Value": {"missing": 0, "unique_nonempty": 17, "numeric_range": [18.0, 36.0]}, "Weight": {"missing": 0, "unique_nonempty": 9, "numeric_range": [5.0, 13.0]}, "LaborHours": {"missing": 0, "unique_nonempty": 10, "numeric_range": [6.0, 15.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "family", "matching_columns": [], "exact_matching_columns": 0}, {"term": "option", "matching_columns": [], "exact_matching_columns": 0}, {"term": "value", "matching_columns": [], "exact_matching_columns": 0}, {"term": "weight", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant22/inputs/resource_limits.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 2
Columns: ['Resource', 'Limit']
Parsed column types: {'Resource': 'object', 'Limit': 'int64'}
Preview only (first 10 rows):
  Resource Limit
    Weight    55
LaborHours    64
Full-file column statistics: {"Resource": {"missing": 0, "unique_nonempty": 2}, "Limit": {"missing": 0, "unique_nonempty": 2, "numeric_range": [55.0, 64.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "weight", "matching_columns": [{"column": "Resource", "exact": 1, "prefix": 1, "contains": 1, "examples": ["Weight"]}], "exact_matching_columns": 1}]