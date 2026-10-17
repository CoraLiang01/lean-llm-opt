File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/customer_demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 12
Columns: ['customer', 'demand']
Parsed column types: {'customer': 'object', 'demand': 'int64'}
Preview only (first 10 rows):
customer demand
      C1     11
      C2   1148
      C3     54
      C4    833
      C5    154
      C6    551
      C7   7081
      C8     76
      C9     66
     C10    174
Full-file column statistics: {"customer": {"missing": 0, "unique_nonempty": 12}, "demand": {"missing": 0, "unique_nonempty": 12, "numeric_range": [11.0, 7081.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer", "matching_columns": [], "exact_matching_columns": 0}, {"term": "customer_demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/supply_capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 11
Columns: ['Unnamed: 0', 'supply_capacity']
Parsed column types: {'Unnamed: 0': 'object', 'supply_capacity': 'int64'}
Preview only (first 10 rows):
Unnamed: 0 supply_capacity
        S1               4
        S2             575
        S3            1504
        S4             178
        S5             228
        S6              50
        S7               3
        S8            6148
        S9               6
       S10           10673
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 11}, "supply_capacity": {"missing": 0, "unique_nonempty": 11, "numeric_range": [3.0, 10673.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer_demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/transportation_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 11
Columns: ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12']
Parsed column types: {'Unnamed: 0': 'object', 'C1': 'float64', 'C2': 'float64', 'C3': 'float64', 'C4': 'float64', 'C5': 'float64', 'C6': 'float64', 'C7': 'float64', 'C8': 'float64', 'C9': 'float64', 'C10': 'float64', 'C11': 'float64', 'C12': 'float64'}
Preview only (first 10 rows):
Unnamed: 0                 C1                 C2                 C3                 C4                 C5                 C6                 C7                 C8                    C9                C10                C11                C12
        S1  0.639144476970582  49.71842803015729  33.75857739960576  1570.673110465785 1370.4095474322417  57.35307774277479  57.18299486453194   54.9209612366192     1143.680909226399  52.49127007738756  606.4434399601076 1192.4686514332489
        S2  605.4786373569875  64.53562572761275  478.4779031378926  887.0480739088434  65.46111249492031  71.93605217833378  41.29015388498019  70.36038207491039     35.35892996332259 1472.7481944140839 0.6004591535232997  49.86854015671519
        S3 1139.0440074582496  4.785056325458736 1805.6214229758102 1302.8958147418275  2437.321229159901 103.80368582531935  774.6558236505713  4.515988277174664     879.7048537066717 162.70556734409897  1208.613484750161 110.18688517926226
        S4  69.26989890601938  2105.485387219297  869.6820232492624 1494.8985656180187  310.5376623181487  98.15455717980421 103.36918486373995 1758.8783768888936     97.28540713798621   94.6504089308906  1277.251451477508 21.636190287574664
        S5  980.4114260089906  899.3108831856057  1183.032552702089  402.0986161964097  81.78864123893297 1115.6819455776936 123.80427864011308   1121.14687497079 0.0024451264081394235 1009.6451734576028 35.348018297991366  1625.434626662855
        S6  1246.782499848912 2105.7966672265507 1014.3393213554372 1494.6681439414933  362.0173906933171  98.17142420905459  2170.405867933639   97.7318771093979     97.26834016110725 1987.9907846635351  70.94396914331519 389.15980447937727
        S7   57.1086015288379 23.836168859030245  78.10572975165614  742.8068113300407 1926.0796823736941  454.3789956981779  458.2901436941235  465.9307664524444    28.138607069878855  524.6154260270081   997.531783848061 104.47794493215576
        S8  981.2908605082814 120.90130000942015 1625.8206931791087 1267.8229294135008 2569.6446053909003 13.471811837256363  815.1525428026629 253.42349641458793     43.76562945456531  275.9784134803488   1228.06989342366 103.48323020673556
        S9 30.532779511898102 1444.8594995969975  173.5547323639261 1307.3913121142912  965.2012304156898 1843.7769498110006 1483.6408846054035  85.32209952688736     1353.500934450796 1485.9153764236357 29.423790844675874 26.619419605630917
       S10  94.11093956131819 1422.9971302244805  1470.776907673336 1419.3382251704456  38.94527784177093  72.20112949102915 2040.4605902793303  1542.702557551204    1803.8001691896025  72.94365832842001  2181.454205937846  973.5515553238279
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 11}, "C1": {"missing": 0, "unique_nonempty": 11, "numeric_range": [0.639144476970582, 1246.782499848912]}, "C2": {"missing": 0, "unique_nonempty": 11, "numeric_range": [4.785056325458736, 2105.7966672265507]}, "C3": {"missing": 0, "unique_nonempty": 11, "numeric_range": [33.75857739960576, 1805.6214229758104]}, "C4": {"missing": 0, "unique_nonempty": 11, "numeric_range": [64.668341345234, 1570.673110465785]}, "C5": {"missing": 0, "unique_nonempty": 11, "numeric_range": [38.94527784177093, 2569.6446053909003]}, "C6": {"missing": 0, "unique_nonempty": 11, "numeric_range": [0.0028957952749427, 1843.7769498110008]}, "C7": {"missing": 0, "unique_nonempty": 11, "numeric_range": [41.29015388498019, 2170.405867933639]}, "C8": {"missing": 0, "unique_nonempty": 11, "numeric_range": [4.515988277174664, 1758.8783768888936]}, "C9": {"missing": 0, "unique_nonempty": 11, "numeric_range": [0.0024451264081394, 1803.8001691896025]}, "C10": {"missing": 0, "unique_nonempty": 11, "numeric_range": [52.49127007738756, 1987.9907846635351]}, "C11": {"missing": 0, "unique_nonempty": 11, "numeric_range": [0.6004591535232997, 2181.454205937846]}, "C12": {"missing": 0, "unique_nonempty": 11, "numeric_range": [21.636190287574664, 2330.76820970791]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer_demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv", "matching_columns": [], "exact_matching_columns": 0}]