File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 13
Columns: ['Task Time Required', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
Parsed column types: {'Task Time Required': 'object', 'A': 'float64', 'B': 'float64', 'C': 'float64', 'D': 'float64', 'E': 'float64', 'F': 'float64', 'G': 'float64', 'H': 'float64', 'I': 'float64', 'J': 'float64'}
Preview only (first 10 rows):
Task Time Required  A B C D E F G H I J
            Worker                     
                 1  9 4 3 7 6 5 6 3 7 5
                 2  4 6 5 6 4 5 3 8 7 6
                 3  5 4 7 5 6 6 5 8 6 9
                 4  7 5 2 3 7 8 5 6 8 5
                 5 10 6 7 4 5 4 4 5 9 7
                 6  6 7 6 3 9 5 7 4 3 4
                 7  8 8 5 9 5 7 5 9 5 3
                 8  7 4 8 8 6 7 5 7 7 7
                 9  5 6 8 7 7 8 7 8 4 5
Full-file column statistics: {"Task Time Required": {"missing": 0, "unique_nonempty": 13}, "A": {"missing": 1, "unique_nonempty": 7, "numeric_range": [4.0, 10.0]}, "B": {"missing": 1, "unique_nonempty": 5, "numeric_range": [4.0, 8.0]}, "C": {"missing": 1, "unique_nonempty": 8, "numeric_range": [2.0, 10.0]}, "D": {"missing": 1, "unique_nonempty": 7, "numeric_range": [3.0, 9.0]}, "E": {"missing": 1, "unique_nonempty": 6, "numeric_range": [4.0, 9.0]}, "F": {"missing": 1, "unique_nonempty": 5, "numeric_range": [4.0, 8.0]}, "G": {"missing": 1, "unique_nonempty": 7, "numeric_range": [3.0, 9.0]}, "H": {"missing": 1, "unique_nonempty": 7, "numeric_range": [3.0, 9.0]}, "I": {"missing": 1, "unique_nonempty": 7, "numeric_range": [3.0, 9.0]}, "J": {"missing": 1, "unique_nonempty": 6, "numeric_range": [3.0, 9.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "a", "matching_columns": [], "exact_matching_columns": 0}, {"term": "worker", "matching_columns": [{"column": "Task Time Required", "exact": 1, "prefix": 1, "contains": 1, "examples": ["Worker"]}], "exact_matching_columns": 1}]
In a worker-task matrix, distinguish embedded row/column-axis captions from actual workers. Select workers by their identifiers, exclude label rows, then validate worker counts; never truncate by position.