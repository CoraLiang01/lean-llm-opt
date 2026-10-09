File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 11
Columns: ['Unnamed: 0', 'Depot', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
Parsed column types: {'Unnamed: 0': 'object', 'Depot': 'int64', 'A': 'int64', 'B': 'int64', 'C': 'int64', 'D': 'int64', 'E': 'int64', 'F': 'int64', 'G': 'int64', 'H': 'int64', 'I': 'int64', 'J': 'int64'}
Preview only (first 10 rows):
Unnamed: 0 Depot  A  B   C  D  E  F  G  H   I  J
     Depot     0 28 41  63 39 38 45 35 28  44 35
         A    28  0 27  87 35 65 63 41 39  43 20
         B    41 27  0  81 13 77 54 25 63  70  7
         C    63 87 81   0 69 53 28 57 83 102 81
         D    39 35 13  69  0 72 41 12 64  75 17
         E    38 65 77  53 72  0 53 64 39  58 72
         F    45 63 54  28 41 53  0 29 70  88 54
         G    35 41 25  57 12 64 29  0 63  76 27
         H    28 39 63  83 64 39 70 63  0  20 56
         I    44 43 70 102 75 58 88 76 20   0 63
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 11}, "Depot": {"missing": 0, "unique_nonempty": 9, "numeric_range": [0.0, 63.0]}, "A": {"missing": 0, "unique_nonempty": 11, "numeric_range": [0.0, 87.0]}, "B": {"missing": 0, "unique_nonempty": 11, "numeric_range": [0.0, 81.0]}, "C": {"missing": 0, "unique_nonempty": 10, "numeric_range": [0.0, 102.0]}, "D": {"missing": 0, "unique_nonempty": 11, "numeric_range": [0.0, 75.0]}, "E": {"missing": 0, "unique_nonempty": 9, "numeric_range": [0.0, 77.0]}, "F": {"missing": 0, "unique_nonempty": 10, "numeric_range": [0.0, 88.0]}, "G": {"missing": 0, "unique_nonempty": 11, "numeric_range": [0.0, 76.0]}, "H": {"missing": 0, "unique_nonempty": 9, "numeric_range": [0.0, 83.0]}, "I": {"missing": 0, "unique_nonempty": 11, "numeric_range": [0.0, 102.0]}, "J": {"missing": 0, "unique_nonempty": 11, "numeric_range": [0.0, 81.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "a", "matching_columns": [{"column": "Unnamed: 0", "exact": 1, "prefix": 1, "contains": 1, "examples": ["A"]}], "exact_matching_columns": 1}, {"term": "b", "matching_columns": [{"column": "Unnamed: 0", "exact": 1, "prefix": 1, "contains": 1, "examples": ["B"]}], "exact_matching_columns": 1}, {"term": "c", "matching_columns": [{"column": "Unnamed: 0", "exact": 1, "prefix": 1, "contains": 1, "examples": ["C"]}], "exact_matching_columns": 1}, {"term": "depot", "matching_columns": [{"column": "Unnamed: 0", "exact": 1, "prefix": 1, "contains": 1, "examples": ["Depot"]}], "exact_matching_columns": 1}]