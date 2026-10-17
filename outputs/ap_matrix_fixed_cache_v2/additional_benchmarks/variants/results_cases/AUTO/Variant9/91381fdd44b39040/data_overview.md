File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 4
Columns: ['Item', 'Demand']
Parsed column types: {'Item': 'object', 'Demand': 'int64'}
Preview only (first 10 rows):
Item Demand
   A     24
   B     18
   C     12
   D     10
Full-file column statistics: {"Item": {"missing": 0, "unique_nonempty": 4}, "Demand": {"missing": 0, "unique_nonempty": 4, "numeric_range": [10.0, 24.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "a", "matching_columns": [{"column": "Item", "exact": 1, "prefix": 1, "contains": 1, "examples": ["A"]}], "exact_matching_columns": 1}, {"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 9
Columns: ['Pattern', 'A', 'B', 'C', 'D']
Parsed column types: {'Pattern': 'object', 'A': 'int64', 'B': 'int64', 'C': 'int64', 'D': 'int64'}
Preview only (first 10 rows):
Pattern A B C D
     P1 4 0 0 0
     P2 0 3 0 0
     P3 0 0 2 0
     P4 0 0 0 2
     P5 2 1 0 0
     P6 1 0 1 0
     P7 0 1 0 1
     P8 1 1 1 0
     P9 2 0 0 1
Full-file column statistics: {"Pattern": {"missing": 0, "unique_nonempty": 9}, "A": {"missing": 0, "unique_nonempty": 4, "numeric_range": [0.0, 4.0]}, "B": {"missing": 0, "unique_nonempty": 3, "numeric_range": [0.0, 3.0]}, "C": {"missing": 0, "unique_nonempty": 3, "numeric_range": [0.0, 2.0]}, "D": {"missing": 0, "unique_nonempty": 3, "numeric_range": [0.0, 2.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "a", "matching_columns": [], "exact_matching_columns": 0}, {"term": "pattern", "matching_columns": [], "exact_matching_columns": 0}]