File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 71
Columns: ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
Parsed column types: {'Product Name': 'object', 'Revenue': 'float64', 'Demand': 'int64', 'Initial Inventory': 'int64'}
Preview only (first 10 rows):
                    Product Name Revenue Demand Initial Inventory
             Adams Group_service 1003.56     10                80
          Anderson-Leach_against  549.92      6                40
        Anderson-Valdez_somebody   169.7     75               560
              Anderson-White_son 1285.81    121               820
              Andrews LLC_matter 1269.71    130               960
               Andrews PLC_enter  805.82    112               850
            Andrews-Martin_build 1134.19    125               950
              Arias-Mendoza_life 1479.23     54               360
           Bennett and Sons_down  139.14     75               580
Bennett, Foster and Moreno_enter  518.67    102               800
Full-file column statistics: {"Product Name": {"missing": 0, "unique_nonempty": 71}, "Revenue": {"missing": 0, "unique_nonempty": 71, "numeric_range": [139.14, 1479.23]}, "Demand": {"missing": 0, "unique_nonempty": 58, "numeric_range": [4.0, 138.0]}, "Initial Inventory": {"missing": 0, "unique_nonempty": 50, "numeric_range": [30.0, 990.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "initial inventory", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revenue", "matching_columns": [], "exact_matching_columns": 0}]