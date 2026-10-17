File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant14/inputs/facility_sites.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 8
Columns: ['Center', 'OpeningCost', 'CoveredDistricts']
Parsed column types: {'Center': 'object', 'OpeningCost': 'int64', 'CoveredDistricts': 'object'}
Preview only (first 10 rows):
Center OpeningCost CoveredDistricts
    B1          11         Z1;Z2;Z5
    B2          14         Z2;Z3;Z6
    B3          10         Z4;Z5;Z8
    B4          13         Z1;Z6;Z7
    B5          16         Z3;Z7;Z9
    B6           9        Z8;Z9;Z10
    B7          12           Z4;Z10
    B8          15         Z5;Z6;Z9
Full-file column statistics: {"Center": {"missing": 0, "unique_nonempty": 8}, "OpeningCost": {"missing": 0, "unique_nonempty": 8, "numeric_range": [9.0, 16.0]}, "CoveredDistricts": {"missing": 0, "unique_nonempty": 8}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: []

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant14/inputs/service_zones.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['Zone']
Parsed column types: {'Zone': 'object'}
Preview only (first 10 rows):
Zone
  Z1
  Z2
  Z3
  Z4
  Z5
  Z6
  Z7
  Z8
  Z9
 Z10
Full-file column statistics: {"Zone": {"missing": 0, "unique_nonempty": 10}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "zone", "matching_columns": [], "exact_matching_columns": 0}]