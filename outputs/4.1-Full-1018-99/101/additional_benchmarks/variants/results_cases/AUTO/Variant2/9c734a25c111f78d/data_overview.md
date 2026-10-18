File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/supplier_capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 6
Columns: ['Supplier', 'SupplyCapacity']
Parsed column types: {'Supplier': 'object', 'SupplyCapacity': 'int64'}
Preview only (first 10 rows):
Supplier SupplyCapacity
      S1            210
      S2            180
      S3            230
      S4            160
      S5            200
      S6            170
Full-file column statistics: {"Supplier": {"missing": 0, "unique_nonempty": 6}, "SupplyCapacity": {"missing": 0, "unique_nonempty": 6, "numeric_range": [160.0, 230.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "supplier", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/customer_demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 12
Columns: ['Customer', 'Demand']
Parsed column types: {'Customer': 'object', 'Demand': 'int64'}
Preview only (first 10 rows):
Customer Demand
      C1     80
      C2    110
      C3     70
      C4     90
      C5    120
      C6     95
      C7     60
      C8     85
      C9    100
     C10     75
Full-file column statistics: {"Customer": {"missing": 0, "unique_nonempty": 12}, "Demand": {"missing": 0, "unique_nonempty": 11, "numeric_range": [60.0, 120.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer", "matching_columns": [], "exact_matching_columns": 0}, {"term": "demand", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_variable_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 6
Columns: ['Supplier', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12']
Parsed column types: {'Supplier': 'object', 'C1': 'int64', 'C2': 'int64', 'C3': 'int64', 'C4': 'int64', 'C5': 'int64', 'C6': 'int64', 'C7': 'int64', 'C8': 'int64', 'C9': 'int64', 'C10': 'int64', 'C11': 'int64', 'C12': 'int64'}
Preview only (first 10 rows):
Supplier C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12
      S1  4  5 32 35 34 36 31 33 37  35  34  36
      S2 33 31  6  5 35 34 36 32 34  37  35  33
      S3 34 36 33 31  4  7 35 37 32  34  36  35
      S4 36 34 35 33 32 35  5  6 37  31  34  36
      S5 35 37 34 36 33 31 32 35  4   5  37  34
      S6 37 35 36 34 35 33 34 32 36  31   6   4
Full-file column statistics: {"Supplier": {"missing": 0, "unique_nonempty": 6}, "C1": {"missing": 0, "unique_nonempty": 6, "numeric_range": [4.0, 37.0]}, "C2": {"missing": 0, "unique_nonempty": 6, "numeric_range": [5.0, 37.0]}, "C3": {"missing": 0, "unique_nonempty": 6, "numeric_range": [6.0, 36.0]}, "C4": {"missing": 0, "unique_nonempty": 6, "numeric_range": [5.0, 36.0]}, "C5": {"missing": 0, "unique_nonempty": 5, "numeric_range": [4.0, 35.0]}, "C6": {"missing": 0, "unique_nonempty": 6, "numeric_range": [7.0, 36.0]}, "C7": {"missing": 0, "unique_nonempty": 6, "numeric_range": [5.0, 36.0]}, "C8": {"missing": 0, "unique_nonempty": 5, "numeric_range": [6.0, 37.0]}, "C9": {"missing": 0, "unique_nonempty": 5, "numeric_range": [4.0, 37.0]}, "C10": {"missing": 0, "unique_nonempty": 5, "numeric_range": [5.0, 37.0]}, "C11": {"missing": 0, "unique_nonempty": 5, "numeric_range": [6.0, 37.0]}, "C12": {"missing": 0, "unique_nonempty": 5, "numeric_range": [4.0, 36.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "supplier", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_fixed_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 6
Columns: ['Supplier', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12']
Parsed column types: {'Supplier': 'object', 'C1': 'int64', 'C2': 'int64', 'C3': 'int64', 'C4': 'int64', 'C5': 'int64', 'C6': 'int64', 'C7': 'int64', 'C8': 'int64', 'C9': 'int64', 'C10': 'int64', 'C11': 'int64', 'C12': 'int64'}
Preview only (first 10 rows):
Supplier   C1   C2   C3   C4   C5   C6   C7   C8   C9  C10  C11  C12
      S1  420  450  980 1050 1010 1080  990 1040 1120 1060 1030 1090
      S2 1020  990  480  460 1070 1040 1100 1010 1050 1130 1080 1000
      S3 1040 1090 1010  980  500  530 1060 1110  990 1030 1100 1060
      S4 1100 1040 1080 1010  980 1050  410  440 1120  970 1030 1090
      S5 1060 1120 1040 1090 1000  970  990 1060  470  430 1130 1040
      S6 1120 1060 1100 1040 1070 1010 1030  980 1090  970  490  420
Full-file column statistics: {"Supplier": {"missing": 0, "unique_nonempty": 6}, "C1": {"missing": 0, "unique_nonempty": 6, "numeric_range": [420.0, 1120.0]}, "C2": {"missing": 0, "unique_nonempty": 6, "numeric_range": [450.0, 1120.0]}, "C3": {"missing": 0, "unique_nonempty": 6, "numeric_range": [480.0, 1100.0]}, "C4": {"missing": 0, "unique_nonempty": 6, "numeric_range": [460.0, 1090.0]}, "C5": {"missing": 0, "unique_nonempty": 5, "numeric_range": [500.0, 1070.0]}, "C6": {"missing": 0, "unique_nonempty": 6, "numeric_range": [530.0, 1080.0]}, "C7": {"missing": 0, "unique_nonempty": 5, "numeric_range": [410.0, 1100.0]}, "C8": {"missing": 0, "unique_nonempty": 6, "numeric_range": [440.0, 1110.0]}, "C9": {"missing": 0, "unique_nonempty": 5, "numeric_range": [470.0, 1120.0]}, "C10": {"missing": 0, "unique_nonempty": 5, "numeric_range": [430.0, 1130.0]}, "C11": {"missing": 0, "unique_nonempty": 5, "numeric_range": [490.0, 1130.0]}, "C12": {"missing": 0, "unique_nonempty": 5, "numeric_range": [420.0, 1090.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "supplier", "matching_columns": [], "exact_matching_columns": 0}]