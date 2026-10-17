File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant4/inputs/service_centers.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 8
Columns: ['Center', 'OpeningCost', 'CoveredDistricts']
Parsed column types: {'Center': 'object', 'OpeningCost': 'int64', 'CoveredDistricts': 'object'}
Preview only (first 10 rows):
Center OpeningCost CoveredDistricts
   SC1          12         D1;D2;D4
   SC2          15         D2;D3;D5
   SC3          18         D4;D5;D6
   SC4          10            D6;D7
   SC5          14        D7;D8;D10
   SC6          13            D8;D9
   SC7          16        D1;D9;D10
   SC8          11         D3;D4;D8
Full-file column statistics: {"Center": {"missing": 0, "unique_nonempty": 8}, "OpeningCost": {"missing": 0, "unique_nonempty": 8, "numeric_range": [10.0, 18.0]}, "CoveredDistricts": {"missing": 0, "unique_nonempty": 8}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "center", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant4/inputs/districts.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['District']
Parsed column types: {'District': 'object'}
Preview only (first 10 rows):
District
      D1
      D2
      D3
      D4
      D5
      D6
      D7
      D8
      D9
     D10
Full-file column statistics: {"District": {"missing": 0, "unique_nonempty": 10}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "district", "matching_columns": [], "exact_matching_columns": 0}]