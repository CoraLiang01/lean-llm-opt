File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant18/inputs/item_demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 5
Columns: ['Item', 'Demand']
Parsed column types: {'Item': 'object', 'Demand': 'int64'}
Preview only (first 10 rows):
Item Demand
   A     18
   B     14
   C     12
   D     10
   E      8
Full-file column statistics: {"Item": {"missing": 0, "unique_nonempty": 5}, "Demand": {"missing": 0, "unique_nonempty": 5, "numeric_range": [8.0, 18.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "a", "matching_columns": [{"column": "Item", "exact": 1, "prefix": 1, "contains": 1, "examples": ["A"]}], "exact_matching_columns": 1}, {"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant18/inputs/cutting_patterns.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['Pattern', 'A', 'B', 'C', 'D', 'E']
Parsed column types: {'Pattern': 'object', 'A': 'int64', 'B': 'int64', 'C': 'int64', 'D': 'int64', 'E': 'int64'}
Preview only (first 10 rows):
Pattern A B C D E
     P1 3 0 0 0 0
     P2 0 2 1 0 0
     P3 0 0 2 1 0
     P4 0 0 0 2 1
     P5 1 1 0 1 0
     P6 2 0 1 0 0
     P7 0 1 1 0 1
     P8 1 0 0 1 1
     P9 1 2 0 0 0
    P10 0 0 1 1 1
Full-file column statistics: {"Pattern": {"missing": 0, "unique_nonempty": 10}, "A": {"missing": 0, "unique_nonempty": 4, "numeric_range": [0.0, 3.0]}, "B": {"missing": 0, "unique_nonempty": 3, "numeric_range": [0.0, 2.0]}, "C": {"missing": 0, "unique_nonempty": 3, "numeric_range": [0.0, 2.0]}, "D": {"missing": 0, "unique_nonempty": 3, "numeric_range": [0.0, 2.0]}, "E": {"missing": 0, "unique_nonempty": 2, "numeric_range": [0.0, 1.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "a", "matching_columns": [], "exact_matching_columns": 0}, {"term": "pattern", "matching_columns": [], "exact_matching_columns": 0}]