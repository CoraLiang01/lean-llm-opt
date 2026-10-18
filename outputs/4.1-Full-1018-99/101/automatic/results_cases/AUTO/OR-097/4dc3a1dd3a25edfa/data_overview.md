File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 120
Columns: ['Option', 'Cost', 'Delta', 'Gamma', 'Vega', 'MaxLong', 'MaxShort']
Parsed column types: {'Option': 'object', 'Cost': 'int64', 'Delta': 'float64', 'Gamma': 'float64', 'Vega': 'float64', 'MaxLong': 'int64', 'MaxShort': 'int64'}
Preview only (first 10 rows):
Option Cost Delta Gamma Vega MaxLong MaxShort
 Opt_1    9 -0.54  0.12  0.1       9      -14
 Opt_2    6  0.51   0.1 0.16       9      -14
 Opt_3   13  0.17  0.02 0.19      10       -7
 Opt_4   10 -0.24  0.03 0.18       7       -5
 Opt_5    7 -0.61  0.14 0.11      12       -9
 Opt_6    9 -0.26  0.09 0.24       5      -13
 Opt_7   12 -0.24  0.01  0.2      10       -5
 Opt_8    5  0.32  0.02 0.16       8       -7
 Opt_9    9  0.19   0.1 0.17       5       -8
Opt_10   13  0.54  0.01 0.13      11       -5
Full-file column statistics: {"Option": {"missing": 0, "unique_nonempty": 120}, "Cost": {"missing": 0, "unique_nonempty": 12, "numeric_range": [3.0, 14.0]}, "Delta": {"missing": 0, "unique_nonempty": 82, "numeric_range": [-0.69, 0.68]}, "Gamma": {"missing": 0, "unique_nonempty": 15, "numeric_range": [0.01, 0.15]}, "Vega": {"missing": 0, "unique_nonempty": 21, "numeric_range": [0.05, 0.25]}, "MaxLong": {"missing": 0, "unique_nonempty": 10, "numeric_range": [5.0, 14.0]}, "MaxShort": {"missing": 0, "unique_nonempty": 10, "numeric_range": [-14.0, -5.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "cost", "matching_columns": [], "exact_matching_columns": 0}, {"term": "delta", "matching_columns": [], "exact_matching_columns": 0}, {"term": "gamma", "matching_columns": [], "exact_matching_columns": 0}, {"term": "maxlong", "matching_columns": [], "exact_matching_columns": 0}, {"term": "maxshort", "matching_columns": [], "exact_matching_columns": 0}, {"term": "option", "matching_columns": [], "exact_matching_columns": 0}, {"term": "vega", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 120
Columns: ['Unnamed: 0', 'Asset_1', 'Asset_2', 'Asset_3', 'Asset_4', 'Asset_5', 'Asset_6']
Parsed column types: {'Unnamed: 0': 'object', 'Asset_1': 'int64', 'Asset_2': 'int64', 'Asset_3': 'int64', 'Asset_4': 'int64', 'Asset_5': 'int64', 'Asset_6': 'int64'}
Preview only (first 10 rows):
Unnamed: 0 Asset_1 Asset_2 Asset_3 Asset_4 Asset_5 Asset_6
     Opt_1       0       0       0       1       0       0
     Opt_2       1       1       0       0       0       0
     Opt_3       0       0       0       0       0       1
     Opt_4       0       0       0       1       0       0
     Opt_5       0       0       1       0       0       0
     Opt_6       1       0       0       0       0       0
     Opt_7       0       0       0       0       1       1
     Opt_8       0       0       1       0       0       0
     Opt_9       0       0       0       1       0       1
    Opt_10       0       0       1       0       0       0
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 120}, "Asset_1": {"missing": 0, "unique_nonempty": 2, "numeric_range": [0.0, 1.0]}, "Asset_2": {"missing": 0, "unique_nonempty": 2, "numeric_range": [0.0, 1.0]}, "Asset_3": {"missing": 0, "unique_nonempty": 2, "numeric_range": [0.0, 1.0]}, "Asset_4": {"missing": 0, "unique_nonempty": 2, "numeric_range": [0.0, 1.0]}, "Asset_5": {"missing": 0, "unique_nonempty": 2, "numeric_range": [0.0, 1.0]}, "Asset_6": {"missing": 0, "unique_nonempty": 2, "numeric_range": [0.0, 1.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: []
Preserve the query's explicit summation scope: aggregate every summed index inside the expression; never replace a summed index with a separate family of constraints.