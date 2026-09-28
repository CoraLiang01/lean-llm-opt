File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture2/41.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 198
Columns: ['Product Name', 'Labor per unit', 'Material per unit', 'Selling Price', 'Variable Cost']
Parsed column types: {'Product Name': 'object', 'Labor per unit': 'float64', 'Material per unit': 'float64', 'Selling Price': 'int64', 'Variable Cost': 'int64'}
Preview only (first 10 rows):
            Product Name Labor per unit Material per unit Selling Price Variable Cost
    Classic Oxford Shirt            3.1               4.2           125            63
Cotton Crew Neck T-Shirt            2.1               3.1            83            41
     French Terry Hoodie            6.5               6.8           195            95
        Denim Snap Shirt            3.4               4.6           132            68
   Jersey V-Neck T-Shirt            2.5               3.4            90            48
     Unisex Jogger Pants            5.9               6.1           185            88
     Flannel Plaid Shirt            3.5                 4           128            65
  Performance Polo Shirt            2.9               3.9           118            57
      Long-sleeve Henley            2.8               3.7            95            50
  Heavyweight Sweatshirt            6.2               6.4           190            92
Full-file column statistics: {"Product Name": {"missing": 0, "unique_nonempty": 198}, "Labor per unit": {"missing": 0, "unique_nonempty": 49, "numeric_range": [2.0, 8.5]}, "Material per unit": {"missing": 0, "unique_nonempty": 51, "numeric_range": [2.9, 8.8]}, "Selling Price": {"missing": 0, "unique_nonempty": 91, "numeric_range": [78.0, 260.0]}, "Variable Cost": {"missing": 0, "unique_nonempty": 89, "numeric_range": [37.0, 160.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "selling price", "matching_columns": [], "exact_matching_columns": 0}, {"term": "variable cost", "matching_columns": [], "exact_matching_columns": 0}]