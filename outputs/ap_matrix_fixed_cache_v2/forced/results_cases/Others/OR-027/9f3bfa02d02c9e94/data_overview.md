File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 23
Columns: ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory']
Parsed column types: {'Sub Category': 'object', 'Revenue': 'float64', 'Demand': 'int64', 'Initial Inventory': 'float64'}
Preview only (first 10 rows):
     Sub Category Revenue  Demand Initial Inventory
     Atta & Flour   165.2  715183         5346490.0
         Biscuits  181.93  924398         6840830.0
    Breads & Buns  189.99 1006220         7425860.0
            Cakes  484.65  924932         6856120.0
          Chicken  207.75  702263         5204970.0
       Chocolates  437.69  994869         7338980.0
          Cookies  315.21 1031871         7682130.0
    Dals & Pulses    47.4  714036         5233710.0
Edible Oil & Ghee   100.8  900971         6680860.0
             Eggs  308.44  774463         5751560.0
Full-file column statistics: {"Sub Category": {"missing": 0, "unique_nonempty": 23}, "Revenue": {"missing": 0, "unique_nonempty": 23, "numeric_range": [47.4, 918.45]}, "Demand": {"missing": 0, "unique_nonempty": 23, "numeric_range": [674626.0, 1419411.0]}, "Initial Inventory": {"missing": 0, "unique_nonempty": 23, "numeric_range": [4983230.0, 10514390.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "initial inventory", "matching_columns": [], "exact_matching_columns": 0}, {"term": "organ", "matching_columns": [{"column": "Sub Category", "exact": 0, "prefix": 3, "contains": 3, "examples": ["Organic Fruits", "Organic Staples", "Organic Vegetables"]}], "exact_matching_columns": 0}, {"term": "revenue", "matching_columns": [], "exact_matching_columns": 0}]