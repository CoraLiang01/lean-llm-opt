File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/customer_demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['customer', 'demand']
Parsed column types: {'customer': 'object', 'demand': 'int64'}
Preview only (first 10 rows):
customer demand
      C1    216
      C2    168
      C3    264
      C4    216
      C5    216
      C6    192
      C7    144
      C8    168
      C9    168
     C10    168
Full-file column statistics: {"customer": {"missing": 0, "unique_nonempty": 10}, "demand": {"missing": 0, "unique_nonempty": 5, "numeric_range": [144.0, 264.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer", "matching_columns": [], "exact_matching_columns": 0}, {"term": "customer_demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/supply_capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['Unnamed: 0', 'supply_capacity']
Parsed column types: {'Unnamed: 0': 'object', 'supply_capacity': 'int64'}
Preview only (first 10 rows):
Unnamed: 0 supply_capacity
        S1             288
        S2             288
        S3             264
        S4             264
        S5             216
        S6             216
        S7             168
        S8             216
        S9             240
       S10             168
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 10}, "supply_capacity": {"missing": 0, "unique_nonempty": 5, "numeric_range": [168.0, 288.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer_demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/transportation_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
Parsed column types: {'Unnamed: 0': 'object', 'C1': 'float64', 'C2': 'float64', 'C3': 'float64', 'C4': 'float64', 'C5': 'float64', 'C6': 'float64', 'C7': 'float64', 'C8': 'float64', 'C9': 'float64', 'C10': 'float64'}
Preview only (first 10 rows):
Unnamed: 0                 C1                 C2                 C3                   C4                  C5                 C6                  C7                   C8                 C9                C10
        S1  590.3648136504455 23.669172607322494  88.89005869765714   497.52228807074613  466.09034321595647 29.022096827063212  23.675244833973835   23.677760288117437 0.3118394914937161 58.895473920714416
        S2 2042.0715001593626  2133.978484314172    705.15912033561   101.59454516295598   2052.937657376311  1738.754951414345  101.61094965654742   101.61062174376057 122.45214268700826  67.29170751036096
        S3 22.297222160217984  497.9271939314995 1653.0828862073263    23.68545123339267  1386.0807887282344  26.13715280763276   497.6220482906461    498.0935847133144  865.3816296318804 1008.6717394620979
        S4  960.7814533858373 49.128300053752405  1324.238697073691   1032.2095478151716 0.07804725392720868 53.308268726049285    49.1364167175093   1031.8214424894484 466.00495307991264 1351.8189071012546
        S5 1471.2721666392908   85.6956072820555  38.89266823851542   1542.0500358120464  112.20514372003504   82.3702016356405  1542.3399196620971    85.69238745806277 1924.9360769614245 1094.6695960752636
        S6  191.9058726130392 158.50401031820448  91.02045349777458   184.44747201726193    968.146798696633  284.1076062070199   8.791061587686942   158.70523835548545 27.943874345249665   929.807168280051
        S7  81.23891457326876 0.3744642223062507   2079.46686537067   0.3065671755503025  1031.7772962191823  7.203964492497209 0.07623072241762692 0.032473879548006554 23.685827966421357  849.9799406578097
        S8 56.099310965461356  935.6143108671334  73.08824617002863    52.00392409272077   4.025792388934198 1002.2327657984296   935.7766029588662    935.7007252277288  612.8698719325438 1348.8366145919845
        S9  4.502283326860296 0.3899585342810754 1782.4662178163346 0.006345906612718274  1031.9910114913148 129.50665619510303  0.2118319573481142    0.645730107353115 497.62723911435927  40.46575554562011
       S10  333.6869270439132  277.4719386113677  86.02096892455509   277.30836609256806  1004.4649084520337 19.950336815857597  13.202073690286834   238.14321521805866  411.0580332361589  941.7526365563969
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 10}, "C1": {"missing": 0, "unique_nonempty": 10, "numeric_range": [4.502283326860296, 2042.0715001593624]}, "C2": {"missing": 0, "unique_nonempty": 10, "numeric_range": [0.3744642223062507, 2133.978484314172]}, "C3": {"missing": 0, "unique_nonempty": 10, "numeric_range": [38.89266823851542, 2079.46686537067]}, "C4": {"missing": 0, "unique_nonempty": 10, "numeric_range": [0.0063459066127182, 1542.0500358120464]}, "C5": {"missing": 0, "unique_nonempty": 10, "numeric_range": [0.0780472539272086, 2052.937657376311]}, "C6": {"missing": 0, "unique_nonempty": 10, "numeric_range": [7.203964492497209, 1738.754951414345]}, "C7": {"missing": 0, "unique_nonempty": 10, "numeric_range": [0.0762307224176269, 1542.3399196620971]}, "C8": {"missing": 0, "unique_nonempty": 10, "numeric_range": [0.0324738795480065, 1031.8214424894484]}, "C9": {"missing": 0, "unique_nonempty": 10, "numeric_range": [0.3118394914937161, 1924.9360769614243]}, "C10": {"missing": 0, "unique_nonempty": 10, "numeric_range": [40.46575554562011, 1351.8189071012546]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer_demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv", "matching_columns": [], "exact_matching_columns": 0}]