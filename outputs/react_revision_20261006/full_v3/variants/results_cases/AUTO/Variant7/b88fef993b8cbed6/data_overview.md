File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant7/inputs/node_supply_demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 7
Columns: ['Node', 'NodeType', 'Amount']
Parsed column types: {'Node': 'object', 'NodeType': 'object', 'Amount': 'int64'}
Preview only (first 10 rows):
Node       NodeType Amount
  S1   SourceSupply    120
  S2   SourceSupply    100
  S3   SourceSupply     90
  C1 CustomerDemand     70
  C2 CustomerDemand     80
  C3 CustomerDemand     60
  C4 CustomerDemand     90
Full-file column statistics: {"Node": {"missing": 0, "unique_nonempty": 7}, "NodeType": {"missing": 0, "unique_nonempty": 2}, "Amount": {"missing": 0, "unique_nonempty": 6, "numeric_range": [60.0, 120.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: []

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant7/inputs/hub_capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 2
Columns: ['Hub', 'ThroughputCapacity']
Parsed column types: {'Hub': 'object', 'ThroughputCapacity': 'int64'}
Preview only (first 10 rows):
Hub ThroughputCapacity
 H1                170
 H2                160
Full-file column statistics: {"Hub": {"missing": 0, "unique_nonempty": 2}, "ThroughputCapacity": {"missing": 0, "unique_nonempty": 2, "numeric_range": [160.0, 170.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "hub", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant7/inputs/arc_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 14
Columns: ['From', 'To', 'Cost']
Parsed column types: {'From': 'object', 'To': 'object', 'Cost': 'int64'}
Preview only (first 10 rows):
From To Cost
  S1 H1    2
  S1 H2    6
  S2 H1    4
  S2 H2    3
  S3 H1    7
  S3 H2    2
  H1 C1    3
  H1 C2    4
  H1 C3    7
  H1 C4    8
Full-file column statistics: {"From": {"missing": 0, "unique_nonempty": 5}, "To": {"missing": 0, "unique_nonempty": 6}, "Cost": {"missing": 0, "unique_nonempty": 6, "numeric_range": [2.0, 8.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "cost", "matching_columns": [], "exact_matching_columns": 0}, {"term": "from", "matching_columns": [], "exact_matching_columns": 0}, {"term": "to", "matching_columns": [], "exact_matching_columns": 0}]