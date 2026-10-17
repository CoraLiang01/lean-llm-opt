File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 12
Columns: ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
Parsed column types: {'Product Name': 'object', 'Revenue': 'float64', 'Demand': 'int64', 'Initial Inventory': 'int64'}
Preview only (first 10 rows):
          Product Name Revenue Demand Initial Inventory
      Baby Food_255.28  255.28 765850           5627060
       Beverages_47.45   47.45 825453           6131330
          Cereal_205.7   205.7 627481           4656850
        Clothes_109.28  109.28 800987           5913850
       Cosmetics_437.2   437.2 718806           5332910
           Fruits_9.33    9.33 798999           5916720
      Household_668.27  668.27 591313           4402490
           Meat_421.89  421.89 713606           5333760
Office Supplies_651.21  651.21 838862           6176410
   Personal Care_81.73   81.73 756433           5604800
Full-file column statistics: {"Product Name": {"missing": 0, "unique_nonempty": 12}, "Revenue": {"missing": 0, "unique_nonempty": 12, "numeric_range": [9.33, 668.27]}, "Demand": {"missing": 0, "unique_nonempty": 12, "numeric_range": [591313.0, 838862.0]}, "Initial Inventory": {"missing": 0, "unique_nonempty": 12, "numeric_range": [4402490.0, 6176410.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "baby", "matching_columns": [{"column": "Product Name", "exact": 0, "prefix": 1, "contains": 1, "examples": ["Baby Food_255.28"]}], "exact_matching_columns": 0}, {"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "initial inventory", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revenue", "matching_columns": [], "exact_matching_columns": 0}]