File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 15
Columns: ['Unnamed: 0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
Parsed column types: {'Unnamed: 0': 'int64', '1': 'float64', '2': 'float64', '3': 'float64', '4': 'float64', '5': 'float64', '6': 'float64', '7': 'float64', '8': 'float64', '9': 'float64', '10': 'float64', '11': 'float64', '12': 'float64', '13': 'float64', '14': 'float64', '15': 'float64'}
Preview only (first 10 rows):
Unnamed: 0 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
         1   67 55 80 21 77 78 74 85 28 55 53 66 89 78
         2      38 29 68 36 62 54 49 92 37 51 38 82 31
         3         28 44 27 56 34 33 68 70 55 46 32 40
         4            21 51 46 48 31 55 68 85 58 56 22
         5               42 57 31 55 79 49 70 43 55 78
         6                  63 41 39 52 76 54 59 44 76
         7                     38 35 37 55 54 51 14 64
         8                        53 24 60 42 31 42 27
         9                           88 28 65 12 63 45
        10                              84 40 43 81 37
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 15, "numeric_range": [1.0, 15.0]}, "1": {"missing": 15, "unique_nonempty": 0}, "2": {"missing": 14, "unique_nonempty": 1, "numeric_range": [67.0, 67.0]}, "3": {"missing": 13, "unique_nonempty": 2, "numeric_range": [38.0, 55.0]}, "4": {"missing": 12, "unique_nonempty": 3, "numeric_range": [28.0, 80.0]}, "5": {"missing": 11, "unique_nonempty": 3, "numeric_range": [21.0, 68.0]}, "6": {"missing": 10, "unique_nonempty": 5, "numeric_range": [27.0, 77.0]}, "7": {"missing": 9, "unique_nonempty": 6, "numeric_range": [46.0, 78.0]}, "8": {"missing": 8, "unique_nonempty": 7, "numeric_range": [31.0, 74.0]}, "9": {"missing": 7, "unique_nonempty": 8, "numeric_range": [31.0, 85.0]}, "10": {"missing": 6, "unique_nonempty": 9, "numeric_range": [24.0, 92.0]}, "11": {"missing": 5, "unique_nonempty": 9, "numeric_range": [28.0, 84.0]}, "12": {"missing": 4, "unique_nonempty": 10, "numeric_range": [40.0, 85.0]}, "13": {"missing": 3, "unique_nonempty": 10, "numeric_range": [12.0, 66.0]}, "14": {"missing": 2, "unique_nonempty": 12, "numeric_range": [14.0, 89.0]}, "15": {"missing": 1, "unique_nonempty": 13, "numeric_range": [22.0, 78.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: []
When the query explicitly states symmetry, align matrix axes by entity ID. Fill a missing off-diagonal entry from its present transpose; preserve zeros. Reject pairs with both entries missing or conflicting values beyond numeric tolerance. Without explicit symmetry, preserve direction and never mirror or zero-fill entries.