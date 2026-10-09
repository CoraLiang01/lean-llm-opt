File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_nodes.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 8
Columns: ['Node']
Parsed column types: {'Node': 'object'}
Preview only (first 10 rows):
Node
  N1
  N2
  N3
  N4
  N5
  N6
  N7
  N8
Full-file column statistics: {"Node": {"missing": 0, "unique_nonempty": 8}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "node", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_edges.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 14
Columns: ['Node1', 'Node2', 'ConstructionCost']
Parsed column types: {'Node1': 'object', 'Node2': 'object', 'ConstructionCost': 'int64'}
Preview only (first 10 rows):
Node1 Node2 ConstructionCost
   N1    N2                4
   N1    N3                3
   N1    N4                9
   N2    N3                5
   N2    N5                6
   N3    N4                4
   N3    N6                7
   N4    N6                2
   N4    N7                8
   N5    N6                3
Full-file column statistics: {"Node1": {"missing": 0, "unique_nonempty": 7}, "Node2": {"missing": 0, "unique_nonempty": 7}, "ConstructionCost": {"missing": 0, "unique_nonempty": 9, "numeric_range": [2.0, 10.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: []

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant10/inputs/network_parameters.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 1
Columns: ['Parameter', 'Value']
Parsed column types: {'Parameter': 'object', 'Value': 'object'}
Preview only (first 10 rows):
Parameter Value
 RootNode    N1
Full-file column statistics: {"Parameter": {"missing": 0, "unique_nonempty": 1}, "Value": {"missing": 0, "unique_nonempty": 1}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: []