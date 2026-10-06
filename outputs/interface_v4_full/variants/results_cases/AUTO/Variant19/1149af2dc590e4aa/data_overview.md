File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant19/inputs/network_nodes.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 9
Columns: ['Node']
Parsed column types: {'Node': 'object'}
Preview only (first 10 rows):
Node
  V1
  V2
  V3
  V4
  V5
  V6
  V7
  V8
  V9
Full-file column statistics: {"Node": {"missing": 0, "unique_nonempty": 9}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "node", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant19/inputs/network_edges.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 16
Columns: ['Node1', 'Node2', 'ConstructionCost']
Parsed column types: {'Node1': 'object', 'Node2': 'object', 'ConstructionCost': 'int64'}
Preview only (first 10 rows):
Node1 Node2 ConstructionCost
   V1    V2                6
   V1    V3                4
   V1    V4                9
   V2    V3                5
   V2    V5                7
   V3    V4                3
   V3    V6                8
   V4    V6                4
   V4    V7               10
   V5    V6                6
Full-file column statistics: {"Node1": {"missing": 0, "unique_nonempty": 8}, "Node2": {"missing": 0, "unique_nonempty": 8}, "ConstructionCost": {"missing": 0, "unique_nonempty": 9, "numeric_range": [3.0, 11.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: []

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant19/inputs/network_parameters.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 1
Columns: ['Parameter', 'Value']
Parsed column types: {'Parameter': 'object', 'Value': 'object'}
Preview only (first 10 rows):
Parameter Value
 RootNode    V1
Full-file column statistics: {"Parameter": {"missing": 0, "unique_nonempty": 1}, "Value": {"missing": 0, "unique_nonempty": 1}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: []