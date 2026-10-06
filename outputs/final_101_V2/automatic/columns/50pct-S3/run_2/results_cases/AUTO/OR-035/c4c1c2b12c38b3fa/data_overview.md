File: /Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S3/Large-scale-or/Others_example/Others4/cost.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 120
Columns: ['Food', 'Calories', 'CaloriesPreviousRecipe', 'CostPreviousQuarter_USD', 'Protein(g)', 'Fat(g)', 'DietPlanPreviousQuarter', 'VitaminC(mg)', 'Cost']
Parsed column types: {'Food': 'object', 'Calories': 'int64', 'CaloriesPreviousRecipe': 'int64', 'CostPreviousQuarter_USD': 'float64', 'Protein(g)': 'float64', 'Fat(g)': 'float64', 'DietPlanPreviousQuarter': 'object', 'VitaminC(mg)': 'float64', 'Cost': 'float64'}
Preview only (first 10 rows):
     Food Calories CaloriesPreviousRecipe CostPreviousQuarter_USD Protein(g) Fat(g) DietPlanPreviousQuarter VitaminC(mg) Cost
  Oatmeal      150                    148                 0.40936          5    2.5                  LowFat            0  0.4
     Milk      120                    137                 0.58570          8      5               HighFiber            0  0.5
      Egg       78                     93                 0.32442          6      5                  LowFat            0  0.3
   Banana      105                    126                0.264925          1    0.4               HighFiber           10 0.25
Grain_001      201                    235                0.359442        3.6      4                  LowFat          0.5 0.38
Grain_002      133                    150                 0.80850        6.4    4.6                Balanced          0.2 0.75
Grain_003      123                    101                 0.75237        3.5    1.9             HighProtein          1.2 0.93
Grain_004      211                    198                0.944586        6.2    3.2                  LowFat          0.4 0.97
Grain_005      120                    111                0.633696        6.8    1.6                Balanced          0.8 0.56
Grain_006      147                    131                0.350559        7.8    2.3                Balanced          0.2 0.33
Full-file column statistics: {"Food": {"missing": 0, "unique_nonempty": 120}, "Calories": {"missing": 0, "unique_nonempty": 89, "numeric_range": [21.0, 244.0]}, "CaloriesPreviousRecipe": {"missing": 0, "unique_nonempty": 98, "numeric_range": [20.0, 286.0]}, "CostPreviousQuarter_USD": {"missing": 0, "unique_nonempty": 120, "numeric_range": [0.11928, 2.670868]}, "Protein(g)": {"missing": 0, "unique_nonempty": 84, "numeric_range": [0.1, 24.9]}, "Fat(g)": {"missing": 0, "unique_nonempty": 68, "numeric_range": [0.0, 14.8]}, "DietPlanPreviousQuarter": {"missing": 0, "unique_nonempty": 4}, "VitaminC(mg)": {"missing": 0, "unique_nonempty": 59, "numeric_range": [0.0, 114.5]}, "Cost": {"missing": 0, "unique_nonempty": 80, "numeric_range": [0.14, 2.37]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "calories", "matching_columns": [], "exact_matching_columns": 0}, {"term": "cost", "matching_columns": [], "exact_matching_columns": 0}, {"term": "food", "matching_columns": [], "exact_matching_columns": 0}]