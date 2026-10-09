File: /Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Others_example/Others4/cost.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 120
Columns: ['Food', 'Calories', 'ScannedPageCount', 'ArchiveRevisionCount', 'Protein(g)', 'Fat(g)', 'ArchiveFolder', 'VitaminC(mg)', 'Cost']
Parsed column types: {'Food': 'object', 'Calories': 'int64', 'ScannedPageCount': 'int64', 'ArchiveRevisionCount': 'int64', 'Protein(g)': 'float64', 'Fat(g)': 'float64', 'ArchiveFolder': 'object', 'VitaminC(mg)': 'float64', 'Cost': 'float64'}
Preview only (first 10 rows):
     Food Calories ScannedPageCount ArchiveRevisionCount Protein(g) Fat(g) ArchiveFolder VitaminC(mg) Cost
  Oatmeal      150               25                    8          5    2.5      Folder_B            0  0.4
     Milk      120               26                   20          8      5      Folder_C            0  0.5
      Egg       78                8                   20          6      5      Folder_C            0  0.3
   Banana      105               13                   27          1    0.4      Folder_A           10 0.25
Grain_001      201                2                   25        3.6      4      Folder_A          0.5 0.38
Grain_002      133                2                    9        6.4    4.6      Folder_C          0.2 0.75
Grain_003      123                9                   12        3.5    1.9      Folder_C          1.2 0.93
Grain_004      211                3                   14        6.2    3.2      Folder_B          0.4 0.97
Grain_005      120               23                   30        6.8    1.6      Folder_A          0.8 0.56
Grain_006      147               12                    1        7.8    2.3      Folder_C          0.2 0.33
Full-file column statistics: {"Food": {"missing": 0, "unique_nonempty": 120}, "Calories": {"missing": 0, "unique_nonempty": 89, "numeric_range": [21.0, 244.0]}, "ScannedPageCount": {"missing": 0, "unique_nonempty": 28, "numeric_range": [1.0, 29.0]}, "ArchiveRevisionCount": {"missing": 0, "unique_nonempty": 30, "numeric_range": [1.0, 30.0]}, "Protein(g)": {"missing": 0, "unique_nonempty": 84, "numeric_range": [0.1, 24.9]}, "Fat(g)": {"missing": 0, "unique_nonempty": 68, "numeric_range": [0.0, 14.8]}, "ArchiveFolder": {"missing": 0, "unique_nonempty": 3}, "VitaminC(mg)": {"missing": 0, "unique_nonempty": 59, "numeric_range": [0.0, 114.5]}, "Cost": {"missing": 0, "unique_nonempty": 80, "numeric_range": [0.14, 2.37]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "calories", "matching_columns": [], "exact_matching_columns": 0}, {"term": "cost", "matching_columns": [], "exact_matching_columns": 0}, {"term": "food", "matching_columns": [], "exact_matching_columns": 0}]