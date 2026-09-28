File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM3/OnlineSalesDataset.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 119
Columns: ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
Parsed column types: {'Product Name': 'object', 'Revenue': 'float64', 'Demand': 'int64', 'Initial Inventory': 'int64'}
Preview only (first 10 rows):
                       Product Name Revenue Demand Initial Inventory
              1984 by George Orwell   19.99      6                20
            Adidas 3-Stripes Shorts   29.99      7                30
       Adidas Essential Track Pants   44.99      5                20
     Adidas FIFA World Cup Football   29.99      4                20
Adidas Originals Superstar Sneakers   79.99      6                20
    Adidas Originals Trefoil Hoodie   64.99      6                20
    Adidas Ultraboost Running Shoes  179.99      3                10
            Adidas Ultraboost Shoes  179.99      3                10
          Amazon Echo Dot (4th Gen)   49.99      5                20
                Amazon Echo Show 10  249.99      2                10
Full-file column statistics: {"Product Name": {"missing": 0, "unique_nonempty": 119}, "Revenue": {"missing": 0, "unique_nonempty": 82, "numeric_range": [9.99, 3899.99]}, "Demand": {"missing": 0, "unique_nonempty": 8, "numeric_range": [2.0, 15.0]}, "Initial Inventory": {"missing": 0, "unique_nonempty": 4, "numeric_range": [10.0, 50.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "initial inventory", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revenue", "matching_columns": [], "exact_matching_columns": 0}]