File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 2
Columns: ['School', 'Capacity']
Parsed column types: {'School': 'object', 'Capacity': 'int64'}
Preview only (first 10 rows):
School Capacity
     I     2028
    II     1560
Full-file column statistics: {"School": {"missing": 0, "unique_nonempty": 2}, "Capacity": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1560.0, 2028.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "school", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 31
Columns: ['Neighborhood', 'Population_White', 'Population_NonWhite']
Parsed column types: {'Neighborhood': 'object', 'Population_White': 'int64', 'Population_NonWhite': 'int64'}
Preview only (first 10 rows):
Neighborhood Population_White Population_NonWhite
         N01               78                  22
         N02               57                  33
         N03               47                  63
         N04               78                  22
         N05               57                  33
         N06               47                  63
         N07               78                  22
         N08               57                  33
         N09               46                  64
         N10               77                  23
Full-file column statistics: {"Neighborhood": {"missing": 0, "unique_nonempty": 31}, "Population_White": {"missing": 0, "unique_nonempty": 7, "numeric_range": [46.0, 78.0]}, "Population_NonWhite": {"missing": 0, "unique_nonempty": 7, "numeric_range": [22.0, 64.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "neighborhood", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 2
Columns: ['School', 'N01', 'N02', 'N03', 'N04', 'N05', 'N06', 'N07', 'N08', 'N09', 'N10', 'N11', 'N12', 'N13', 'N14', 'N15', 'N16', 'N17', 'N18', 'N19', 'N20', 'N21', 'N22', 'N23', 'N24', 'N25', 'N26', 'N27', 'N28', 'N29', 'N30', 'N31']
Parsed column types: {'School': 'object', 'N01': 'float64', 'N02': 'float64', 'N03': 'float64', 'N04': 'float64', 'N05': 'float64', 'N06': 'float64', 'N07': 'float64', 'N08': 'float64', 'N09': 'float64', 'N10': 'float64', 'N11': 'float64', 'N12': 'float64', 'N13': 'float64', 'N14': 'float64', 'N15': 'float64', 'N16': 'float64', 'N17': 'float64', 'N18': 'float64', 'N19': 'float64', 'N20': 'float64', 'N21': 'float64', 'N22': 'float64', 'N23': 'float64', 'N24': 'float64', 'N25': 'float64', 'N26': 'float64', 'N27': 'float64', 'N28': 'float64', 'N29': 'float64', 'N30': 'float64', 'N31': 'float64'}
Preview only (first 10 rows):
School  N01  N02  N03  N04  N05  N06  N07  N08                N09 N10  N11  N12  N13  N14  N15  N16  N17  N18  N19  N20  N21  N22  N23  N24                N25 N26  N27  N28  N29  N30  N31
     I 1.25  1.3 1.35  1.4 1.45  1.5 1.55  1.6               1.65 1.7 1.75  1.8 1.85  1.9 1.95  2.0 3.08 3.16 3.24 3.32  3.4 3.48 3.56 3.64 3.7199999999999998 3.8 3.88 3.96 4.04 4.12  4.2
    II 3.08 3.16 3.24 3.32  3.4 3.48 3.56 3.64 3.7199999999999998 3.8 3.88 3.96 4.04 4.12  4.2 4.28 1.25  1.3 1.35  1.4 1.45  1.5 1.55  1.6               1.65 1.7 1.75  1.8 1.85  1.9 1.95
Full-file column statistics: {"School": {"missing": 0, "unique_nonempty": 2}, "N01": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.25, 3.08]}, "N02": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.3, 3.16]}, "N03": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.35, 3.24]}, "N04": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.4, 3.32]}, "N05": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.45, 3.4]}, "N06": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.5, 3.48]}, "N07": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.55, 3.56]}, "N08": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.6, 3.64]}, "N09": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.65, 3.72]}, "N10": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.7, 3.8]}, "N11": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.75, 3.88]}, "N12": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.8, 3.96]}, "N13": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.85, 4.04]}, "N14": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.9, 4.12]}, "N15": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.95, 4.2]}, "N16": {"missing": 0, "unique_nonempty": 2, "numeric_range": [2.0, 4.28]}, "N17": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.25, 3.08]}, "N18": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.3, 3.16]}, "N19": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.35, 3.24]}, "N20": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.4, 3.32]}, "N21": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.45, 3.4]}, "N22": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.5, 3.48]}, "N23": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.55, 3.56]}, "N24": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.6, 3.64]}, "N25": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.65, 3.72]}, "N26": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.7, 3.8]}, "N27": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.75, 3.88]}, "N28": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.8, 3.96]}, "N29": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.85, 4.04]}, "N30": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.9, 4.12]}, "N31": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.95, 4.2]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "school", "matching_columns": [], "exact_matching_columns": 0}]