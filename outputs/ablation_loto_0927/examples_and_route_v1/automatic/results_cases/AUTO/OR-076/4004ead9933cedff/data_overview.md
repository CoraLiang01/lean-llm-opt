File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/cost.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['Warehouse ID', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20']
Parsed column types: {'Warehouse ID': 'object', 'C1': 'int64', 'C2': 'int64', 'C3': 'int64', 'C4': 'int64', 'C5': 'int64', 'C6': 'int64', 'C7': 'int64', 'C8': 'int64', 'C9': 'int64', 'C10': 'int64', 'C11': 'int64', 'C12': 'int64', 'C13': 'int64', 'C14': 'int64', 'C15': 'int64', 'C16': 'int64', 'C17': 'int64', 'C18': 'int64', 'C19': 'int64', 'C20': 'int64'}
Preview only (first 10 rows):
Warehouse ID C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 C16 C17 C18 C19 C20
          W1 10 15 20 11 16 18  7 12 22   9  14  19  25  13  17   6  21  15   8  10
          W2 18 12  9 14 10  5 19 23 11  16  20   8  15  22   7  13  24  17  12   6
          W3 13 17 15  8 12 21 16 10  5  24  13  22   7  19  14  18   9  25  11  16
          W4  7 22 11 16 20  8 15 19 13  25   6  14  21   9  23  17  10  18  24   5
          W5 16  9 25 13  7 10 23 14 18  21   5  17   9  24  12  20   6  15  19  11
          W6 22  6 14 19 23 11  8 17  9  12  15  24   5  20  10  25  13   7  18  16
          W7  8 25 17  9 14 22 11  6 16  20  18  13  24   5  19  12  23  10   7  15
          W8 19 11  7 21 15 24 13 16 20   8  17  10  12  23   5  14  22   9  16  25
          W9 12 20  5 23 17 14  9 25 18  11  16  21  10   7  24  15  19   6  13  22
         W10 25 14 22  5 19 12 24  7 15  17  23   6  16  10  20   9  18  11  25  14
Full-file column statistics: {"Warehouse ID": {"missing": 0, "unique_nonempty": 10}, "C1": {"missing": 0, "unique_nonempty": 10, "numeric_range": [7.0, 25.0]}, "C2": {"missing": 0, "unique_nonempty": 10, "numeric_range": [6.0, 25.0]}, "C3": {"missing": 0, "unique_nonempty": 10, "numeric_range": [5.0, 25.0]}, "C4": {"missing": 0, "unique_nonempty": 10, "numeric_range": [5.0, 23.0]}, "C5": {"missing": 0, "unique_nonempty": 10, "numeric_range": [7.0, 23.0]}, "C6": {"missing": 0, "unique_nonempty": 10, "numeric_range": [5.0, 24.0]}, "C7": {"missing": 0, "unique_nonempty": 10, "numeric_range": [7.0, 24.0]}, "C8": {"missing": 0, "unique_nonempty": 10, "numeric_range": [6.0, 25.0]}, "C9": {"missing": 0, "unique_nonempty": 9, "numeric_range": [5.0, 22.0]}, "C10": {"missing": 0, "unique_nonempty": 10, "numeric_range": [8.0, 25.0]}, "C11": {"missing": 0, "unique_nonempty": 10, "numeric_range": [5.0, 23.0]}, "C12": {"missing": 0, "unique_nonempty": 10, "numeric_range": [6.0, 24.0]}, "C13": {"missing": 0, "unique_nonempty": 10, "numeric_range": [5.0, 25.0]}, "C14": {"missing": 0, "unique_nonempty": 10, "numeric_range": [5.0, 24.0]}, "C15": {"missing": 0, "unique_nonempty": 10, "numeric_range": [5.0, 24.0]}, "C16": {"missing": 0, "unique_nonempty": 10, "numeric_range": [6.0, 25.0]}, "C17": {"missing": 0, "unique_nonempty": 10, "numeric_range": [6.0, 24.0]}, "C18": {"missing": 0, "unique_nonempty": 9, "numeric_range": [6.0, 25.0]}, "C19": {"missing": 0, "unique_nonempty": 10, "numeric_range": [7.0, 25.0]}, "C20": {"missing": 0, "unique_nonempty": 9, "numeric_range": [5.0, 25.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: []

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/warehouse.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['Warehouse ID', 'Fixed_Cost', 'Capacity']
Parsed column types: {'Warehouse ID': 'object', 'Fixed_Cost': 'int64', 'Capacity': 'int64'}
Preview only (first 10 rows):
Warehouse ID Fixed_Cost Capacity
          W1       2000     1000
          W2       2500     1500
          W3       1800     1200
          W4       3200     2000
          W5       1500      800
          W6       4000     2500
          W7       2800     1800
          W8       1950     1100
          W9       3500     2100
         W10       2200     1300
Full-file column statistics: {"Warehouse ID": {"missing": 0, "unique_nonempty": 10}, "Fixed_Cost": {"missing": 0, "unique_nonempty": 10, "numeric_range": [1500.0, 4000.0]}, "Capacity": {"missing": 0, "unique_nonempty": 10, "numeric_range": [800.0, 2500.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 20
Columns: ['Customer ID', 'Demand']
Parsed column types: {'Customer ID': 'object', 'Demand': 'int64'}
Preview only (first 10 rows):
Customer ID Demand
         C1    800
         C2    600
         C3    500
         C4    700
         C5    450
         C6    950
         C7    350
         C8    850
         C9    400
        C10    750
Full-file column statistics: {"Customer ID": {"missing": 0, "unique_nonempty": 20}, "Demand": {"missing": 0, "unique_nonempty": 20, "numeric_range": [320.0, 950.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}]