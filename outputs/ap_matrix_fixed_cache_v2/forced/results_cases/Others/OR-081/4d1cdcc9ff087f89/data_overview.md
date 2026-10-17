File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 120
Columns: ['Food', 'Calories', 'Protein(g)', 'Fat(g)', 'VitaminC(mg)', 'Cost']
Parsed column types: {'Food': 'object', 'Calories': 'int64', 'Protein(g)': 'float64', 'Fat(g)': 'float64', 'VitaminC(mg)': 'float64', 'Cost': 'float64'}
Preview only (first 10 rows):
     Food Calories Protein(g) Fat(g) VitaminC(mg) Cost
  Oatmeal      150          5    2.5            0  0.4
     Milk      120          8      5            0  0.5
      Egg       78          6      5            0  0.3
   Banana      105          1    0.4           10 0.25
Grain_001      201        3.6      4          0.5 0.38
Grain_002      133        6.4    4.6          0.2 0.75
Grain_003      123        3.5    1.9          1.2 0.93
Grain_004      211        6.2    3.2          0.4 0.97
Grain_005      120        6.8    1.6          0.8 0.56
Grain_006      147        7.8    2.3          0.2 0.33
Full-file column statistics: {"Food": {"missing": 0, "unique_nonempty": 120}, "Calories": {"missing": 0, "unique_nonempty": 89, "numeric_range": [21.0, 244.0]}, "Protein(g)": {"missing": 0, "unique_nonempty": 84, "numeric_range": [0.1, 24.9]}, "Fat(g)": {"missing": 0, "unique_nonempty": 68, "numeric_range": [0.0, 14.8]}, "VitaminC(mg)": {"missing": 0, "unique_nonempty": 59, "numeric_range": [0.0, 114.5]}, "Cost": {"missing": 0, "unique_nonempty": 80, "numeric_range": [0.14, 2.37]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "calories", "matching_columns": [], "exact_matching_columns": 0}, {"term": "cost", "matching_columns": [], "exact_matching_columns": 0}, {"term": "food", "matching_columns": [], "exact_matching_columns": 0}]