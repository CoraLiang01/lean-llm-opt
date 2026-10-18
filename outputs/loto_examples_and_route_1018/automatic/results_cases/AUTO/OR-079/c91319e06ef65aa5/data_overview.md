File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/facility_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 15
Columns: ['Facility', 'FixedCost', 'Capacity']
Parsed column types: {'Facility': 'object', 'FixedCost': 'int64', 'Capacity': 'int64'}
Preview only (first 10 rows):
Facility FixedCost Capacity
      A1         0       30
      A2       175       10
      A3       300       20
      A4       375       30
      A5       500       40
      A6       200       20
      A7       260       25
      A8       220       30
      A9       320       35
     A10       280       20
Full-file column statistics: {"Facility": {"missing": 0, "unique_nonempty": 15}, "FixedCost": {"missing": 0, "unique_nonempty": 15, "numeric_range": [0.0, 560.0]}, "Capacity": {"missing": 0, "unique_nonempty": 8, "numeric_range": [10.0, 50.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "a1", "matching_columns": [{"column": "Facility", "exact": 1, "prefix": 7, "contains": 7, "examples": ["A1", "A10", "A11"]}], "exact_matching_columns": 1}, {"term": "a15", "matching_columns": [{"column": "Facility", "exact": 1, "prefix": 1, "contains": 1, "examples": ["A15"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/shipping_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 15
Columns: ['Origin', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
Parsed column types: {'Origin': 'object', 'B1': 'int64', 'B2': 'int64', 'B3': 'int64', 'B4': 'int64', 'B5': 'int64', 'B6': 'int64', 'B7': 'int64', 'B8': 'int64'}
Preview only (first 10 rows):
Origin B1 B2 B3 B4 B5 B6 B7 B8
    A1  8  4  3  6  7  5  9  8
    A2  5  2  3  5  6  4  7  6
    A3  4  3  4  6  5  5  6  7
    A4  9  7  5  8  9  6 10  7
    A5 10  4  2  6  8  5  7  3
    A6  6  5  4  5  7  6  8  5
    A7  7  6  5  4  6  7  9  6
    A8  5  4  6  3  5  6  7  6
    A9  8  7  6  7  9  8 10  7
   A10  6  5  7  4  6  5  7  5
Full-file column statistics: {"Origin": {"missing": 0, "unique_nonempty": 15}, "B1": {"missing": 0, "unique_nonempty": 7, "numeric_range": [4.0, 10.0]}, "B2": {"missing": 0, "unique_nonempty": 6, "numeric_range": [2.0, 7.0]}, "B3": {"missing": 0, "unique_nonempty": 6, "numeric_range": [2.0, 7.0]}, "B4": {"missing": 0, "unique_nonempty": 6, "numeric_range": [3.0, 8.0]}, "B5": {"missing": 0, "unique_nonempty": 5, "numeric_range": [5.0, 9.0]}, "B6": {"missing": 0, "unique_nonempty": 5, "numeric_range": [4.0, 8.0]}, "B7": {"missing": 0, "unique_nonempty": 5, "numeric_range": [6.0, 10.0]}, "B8": {"missing": 0, "unique_nonempty": 6, "numeric_range": [3.0, 8.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "a1", "matching_columns": [{"column": "Origin", "exact": 1, "prefix": 7, "contains": 7, "examples": ["A1", "A10", "A11"]}], "exact_matching_columns": 1}, {"term": "a15", "matching_columns": [{"column": "Origin", "exact": 1, "prefix": 1, "contains": 1, "examples": ["A15"]}], "exact_matching_columns": 1}, {"term": "b1", "matching_columns": [], "exact_matching_columns": 0}, {"term": "b8", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/demand_requirements.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 8
Columns: ['Destination', 'Demand']
Parsed column types: {'Destination': 'object', 'Demand': 'int64'}
Preview only (first 10 rows):
Destination Demand
         B1     30
         B2     25
         B3     20
         B4     35
         B5     25
         B6     30
         B7     25
         B8     30
Full-file column statistics: {"Destination": {"missing": 0, "unique_nonempty": 8}, "Demand": {"missing": 0, "unique_nonempty": 4, "numeric_range": [20.0, 35.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "b1", "matching_columns": [{"column": "Destination", "exact": 1, "prefix": 1, "contains": 1, "examples": ["B1"]}], "exact_matching_columns": 1}, {"term": "b8", "matching_columns": [{"column": "Destination", "exact": 1, "prefix": 1, "contains": 1, "examples": ["B8"]}], "exact_matching_columns": 1}, {"term": "demand", "matching_columns": [], "exact_matching_columns": 0}]