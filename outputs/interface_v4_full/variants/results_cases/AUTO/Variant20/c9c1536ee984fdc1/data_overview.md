File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant20/inputs/sensor_sites.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 9
Columns: ['Center', 'OpeningCost', 'CoveredDistricts']
Parsed column types: {'Center': 'object', 'OpeningCost': 'int64', 'CoveredDistricts': 'object'}
Preview only (first 10 rows):
Center OpeningCost CoveredDistricts
    S1           8         Q1;Q2;Q4
    S2          12         Q2;Q3;Q5
    S3          11         Q4;Q6;Q7
    S4          10            Q5;Q8
    S5          13         Q6;Q8;Q9
    S6           9       Q7;Q10;Q11
    S7          14        Q1;Q9;Q10
    S8           7        Q3;Q5;Q11
    S9          15        Q2;Q6;Q10
Full-file column statistics: {"Center": {"missing": 0, "unique_nonempty": 9}, "OpeningCost": {"missing": 0, "unique_nonempty": 9, "numeric_range": [7.0, 15.0]}, "CoveredDistricts": {"missing": 0, "unique_nonempty": 9}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: []

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant20/inputs/monitoring_zones.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 11
Columns: ['Zone']
Parsed column types: {'Zone': 'object'}
Preview only (first 10 rows):
Zone
  Q1
  Q2
  Q3
  Q4
  Q5
  Q6
  Q7
  Q8
  Q9
 Q10
Full-file column statistics: {"Zone": {"missing": 0, "unique_nonempty": 11}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "zone", "matching_columns": [], "exact_matching_columns": 0}]