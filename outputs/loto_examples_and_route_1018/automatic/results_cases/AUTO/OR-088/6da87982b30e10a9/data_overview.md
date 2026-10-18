File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/cost.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 15
Columns: ['plant', 'fixed_cost', 'capacity', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15']
Parsed column types: {'plant': 'object', 'fixed_cost': 'int64', 'capacity': 'int64', 'C1': 'float64', 'C2': 'float64', 'C3': 'float64', 'C4': 'float64', 'C5': 'float64', 'C6': 'float64', 'C7': 'float64', 'C8': 'float64', 'C9': 'float64', 'C10': 'float64', 'C11': 'float64', 'C12': 'float64', 'C13': 'float64', 'C14': 'float64', 'C15': 'float64'}
Preview only (first 10 rows):
plant fixed_cost capacity  C1  C2  C3  C4  C5  C6  C7  C8  C9 C10 C11 C12 C13 C14 C15
   F1      11250      101 7.8 7.6 6.7 7.9 8.1 8.3 7.3 8.2 8.1 8.2 7.3 7.7 6.7 7.1 7.9
   F2      13480      124 5.3   6   5 6.4 5.9 6.2 5.6 6.1 6.3 6.1   5 5.6 5.3 4.9 6.3
   F3      14870      139 7.2 8.1 7.4 8.8 8.5 8.7 7.7 8.7 8.9 8.5 7.2 7.7 7.1 7.6 8.4
   F4      10290       86   7 7.1 6.5 7.9 7.4 7.7 6.7 7.9 7.8 7.3 6.8   7 6.5 6.7 7.6
   F5      16740      157 3.5 3.8 2.9 4.3 3.6 3.9 3.2 4.3 4.5   4 3.2   4 2.9 3.4 3.9
   F6      13960      133 8.2 8.6 7.9 9.5 8.5 9.3 8.5 9.4   9 9.2 8.1 8.7 7.9 8.5   9
   F7      12680      118 6.9 7.6 6.8 8.4   8   8 7.6   8 8.1 7.8 6.9 7.1   7 6.9 7.5
   F8      17890      162 6.9 7.8 7.1 8.7 8.6 8.2 7.2 7.9 8.4 7.9   7 7.4 6.8 7.3   8
   F9      10950       92 3.5 3.8 2.8 4.4 4.2 4.8 3.8   5 4.5 4.1 3.2 3.7 3.7 3.2 4.5
  F10      15320      144 5.2 6.1 5.1 6.3 6.1   6 5.6 6.5 6.2 5.9 5.3 6.1 5.1 5.2 6.2
Full-file column statistics: {"plant": {"missing": 0, "unique_nonempty": 15}, "fixed_cost": {"missing": 0, "unique_nonempty": 15, "numeric_range": [10290.0, 17890.0]}, "capacity": {"missing": 0, "unique_nonempty": 15, "numeric_range": [85.0, 162.0]}, "C1": {"missing": 0, "unique_nonempty": 11, "numeric_range": [3.5, 8.2]}, "C2": {"missing": 0, "unique_nonempty": 12, "numeric_range": [3.8, 8.7]}, "C3": {"missing": 0, "unique_nonempty": 14, "numeric_range": [2.8, 7.9]}, "C4": {"missing": 0, "unique_nonempty": 14, "numeric_range": [4.3, 9.5]}, "C5": {"missing": 0, "unique_nonempty": 12, "numeric_range": [3.6, 8.6]}, "C6": {"missing": 0, "unique_nonempty": 15, "numeric_range": [3.9, 9.3]}, "C7": {"missing": 0, "unique_nonempty": 12, "numeric_range": [3.2, 8.5]}, "C8": {"missing": 0, "unique_nonempty": 12, "numeric_range": [4.3, 9.4]}, "C9": {"missing": 0, "unique_nonempty": 13, "numeric_range": [4.5, 9.3]}, "C10": {"missing": 0, "unique_nonempty": 14, "numeric_range": [4.0, 9.2]}, "C11": {"missing": 0, "unique_nonempty": 13, "numeric_range": [3.2, 8.1]}, "C12": {"missing": 0, "unique_nonempty": 14, "numeric_range": [3.7, 8.7]}, "C13": {"missing": 0, "unique_nonempty": 14, "numeric_range": [2.9, 7.9]}, "C14": {"missing": 0, "unique_nonempty": 13, "numeric_range": [3.2, 8.5]}, "C15": {"missing": 0, "unique_nonempty": 14, "numeric_range": [3.9, 9.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "c1", "matching_columns": [], "exact_matching_columns": 0}, {"term": "c15", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "f1", "matching_columns": [{"column": "plant", "exact": 1, "prefix": 7, "contains": 7, "examples": ["F1", "F10", "F11"]}], "exact_matching_columns": 1}, {"term": "f15", "matching_columns": [{"column": "plant", "exact": 1, "prefix": 1, "contains": 1, "examples": ["F15"]}], "exact_matching_columns": 1}, {"term": "fixed_cost", "matching_columns": [], "exact_matching_columns": 0}, {"term": "plant", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 15
Columns: ['customer', 'demand']
Parsed column types: {'customer': 'object', 'demand': 'int64'}
Preview only (first 10 rows):
customer demand
      C1     83
      C2     76
      C3     91
      C4     68
      C5    104
      C6     97
      C7     88
      C8     73
      C9    109
     C10     95
Full-file column statistics: {"customer": {"missing": 0, "unique_nonempty": 15}, "demand": {"missing": 0, "unique_nonempty": 15, "numeric_range": [67.0, 113.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "c1", "matching_columns": [{"column": "customer", "exact": 1, "prefix": 7, "contains": 7, "examples": ["C1", "C10", "C11"]}], "exact_matching_columns": 1}, {"term": "c15", "matching_columns": [{"column": "customer", "exact": 1, "prefix": 1, "contains": 1, "examples": ["C15"]}], "exact_matching_columns": 1}, {"term": "customer", "matching_columns": [], "exact_matching_columns": 0}, {"term": "demand", "matching_columns": [], "exact_matching_columns": 0}]