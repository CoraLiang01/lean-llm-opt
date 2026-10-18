File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM22/DairyGoodsSalesDataset.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 40
Columns: ['Full_Product_Name', 'Revenue', 'Demand', 'Initial Inventory']
Parsed column types: {'Full_Product_Name': 'object', 'Revenue': 'float64', 'Demand': 'int64', 'Initial Inventory': 'int64'}
Preview only (first 10 rows):
          Full_Product_Name Revenue Demand Initial Inventory
                Butter_Amul   96.86  34102             29862
        Butter_Mother Dairy   48.01  36579             29898
    Butter_Parag Milk Foods    8.83  36086             25208
              Butter_Warana   92.96  41254             30816
            Buttermilk_Amul   40.75  29876             19925
    Buttermilk_Mother Dairy   83.07  41229             26482
             Buttermilk_Raj   15.64  35354             30865
           Buttermilk_Sudha   56.57  29649             33517
                Cheese_Amul  100.74  38558             30929
Cheese_Britannia Industries   28.92  28603             21405
Full-file column statistics: {"Full_Product_Name": {"missing": 0, "unique_nonempty": 40}, "Revenue": {"missing": 0, "unique_nonempty": 40, "numeric_range": [8.69, 100.74]}, "Demand": {"missing": 0, "unique_nonempty": 40, "numeric_range": [28603.0, 45762.0]}, "Initial Inventory": {"missing": 0, "unique_nonempty": 40, "numeric_range": [19925.0, 34914.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "initial inventory", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revenue", "matching_columns": [], "exact_matching_columns": 0}]