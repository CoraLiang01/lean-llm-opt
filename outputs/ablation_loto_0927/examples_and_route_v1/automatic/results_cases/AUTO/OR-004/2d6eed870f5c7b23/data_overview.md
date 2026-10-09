File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/customer_demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 12
Columns: ['customer', 'demand']
Parsed column types: {'customer': 'object', 'demand': 'int64'}
Preview only (first 10 rows):
customer demand
      C1     52
      C2     80
      C3    392
      C4    103
      C5     32
      C6   1426
      C7   1024
      C8   2736
      C9   1129
     C10    676
Full-file column statistics: {"customer": {"missing": 0, "unique_nonempty": 12}, "demand": {"missing": 0, "unique_nonempty": 12, "numeric_range": [31.0, 2736.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer", "matching_columns": [], "exact_matching_columns": 0}, {"term": "customer_demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/supply_capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 12
Columns: ['Unnamed: 0', 'supply_capacity']
Parsed column types: {'Unnamed: 0': 'object', 'supply_capacity': 'int64'}
Preview only (first 10 rows):
Unnamed: 0 supply_capacity
        S1              58
        S2              32
        S3            6161
        S4               4
        S5              47
        S6             178
        S7             142
        S8             164
        S9            1011
       S10               6
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 12}, "supply_capacity": {"missing": 0, "unique_nonempty": 12, "numeric_range": [4.0, 7081.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer_demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/transportation_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 12
Columns: ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12']
Parsed column types: {'Unnamed: 0': 'object', 'C1': 'float64', 'C2': 'float64', 'C3': 'float64', 'C4': 'float64', 'C5': 'float64', 'C6': 'float64', 'C7': 'float64', 'C8': 'float64', 'C9': 'float64', 'C10': 'float64', 'C11': 'float64', 'C12': 'float64'}
Preview only (first 10 rows):
Unnamed: 0                 C1                 C2                 C3                 C4                 C5                 C6                C7                 C8                   C9                C10                C11                C12
        S1 134.72882437877243  72.30045141110347   37.9759927355347    611.84656497826 1650.3353902326157  32.97044486689555 34.99837373425947  73.25207370031997     165.752089835103  82.52035827662691  1538.830198350349 2320.1146477101956
        S2 23.835369186583733 1128.6332865368709 187.05459590556023 1.6077337129635412 2227.7036991312493  72.44693148708649 12.60074797346823 1078.7729548111274    383.7220569377865   82.8115250443967 1702.1724763823142 1925.8446970545492
        S3  1138.498759622659  231.8983422482863   44.1568226726957  962.5481640749796  980.1306032437444 1107.5157506568996 741.7073424442746 136.70204401709762   1182.5161115443857  664.1846792537777  36.70238978444623 1405.0506919253128
        S4 1043.7562803117191 24.207016063238527 1120.9890438994094 1027.5437638768522  893.4457577903337 1244.2626779205734 980.0297707948428  513.5650272201696    977.6321548115833  642.5520451224766  437.9556823253786  76.12498668603241
        S5  4.939930661779214 1278.1815735383598  549.7570743190075 21.335474111511868  98.32689279702352  452.0903462786745 595.5981933612871  70.84029746549007 0.001236242263106387 1783.0865677438942 1619.6154655942953 116.08497905117572
        S6 2105.5398596409113 1340.9770559580859 2077.2747284488696 2202.2515396165613   20.5336958780729 2514.0983598337716 2393.213466538681  1197.765543404598   102.32440248566687  788.4640306393768  818.9532260267922 17.249645794282493
        S7  61.36183063260923 113.23716964851326 50.231076475261865 1219.1951077346878   869.868986439361   58.3717723017115 957.3213131437595  168.9964798230943    1363.342713601386  519.3858364273143  483.3345987058458 1412.7652882009972
        S8 1169.3640988900754 1037.2170744709088  732.3577706907058  865.2255769415611 1510.0296187049466   780.853789775776 860.7971696230202  935.9126432821531    61.82504909829045  72.78392607232098 1479.1137195486635  65.87463475119084
        S9  7.936521667214095 1357.5387610434743  628.3001825422914 25.245141819018265 1760.0085249934855  604.6583535734275  696.677483277236 1586.4093183806565    89.68006942976479 1580.7262556456867 1423.5615412405277 2000.5637996766272
       S10 1685.3409758586952 437.39384912973503 1568.8939936485654 1486.9268967582923 498.37544466423503 1493.3860214089566 70.17952878640685   526.999992252258   1527.6606596831157  2.676413573774264 202.72671236450077  45.95308313010429
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 12}, "C1": {"missing": 0, "unique_nonempty": 12, "numeric_range": [4.939930661779214, 2105.539859640912]}, "C2": {"missing": 0, "unique_nonempty": 12, "numeric_range": [24.207016063238527, 1357.5387610434743]}, "C3": {"missing": 0, "unique_nonempty": 12, "numeric_range": [37.9759927355347, 2077.2747284488696]}, "C4": {"missing": 0, "unique_nonempty": 12, "numeric_range": [1.6077337129635412, 2202.2515396165613]}, "C5": {"missing": 0, "unique_nonempty": 12, "numeric_range": [20.5336958780729, 2227.7036991312493]}, "C6": {"missing": 0, "unique_nonempty": 12, "numeric_range": [18.59349093795307, 2514.098359833772]}, "C7": {"missing": 0, "unique_nonempty": 12, "numeric_range": [12.60074797346823, 2393.213466538681]}, "C8": {"missing": 0, "unique_nonempty": 12, "numeric_range": [70.84029746549007, 1586.4093183806565]}, "C9": {"missing": 0, "unique_nonempty": 12, "numeric_range": [0.0012362422631063, 1782.8442633270377]}, "C10": {"missing": 0, "unique_nonempty": 12, "numeric_range": [0.0104862180150152, 1783.0865677438942]}, "C11": {"missing": 0, "unique_nonempty": 12, "numeric_range": [11.1693204455916, 1702.1724763823142]}, "C12": {"missing": 0, "unique_nonempty": 12, "numeric_range": [17.249645794282493, 2320.114647710196]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer_demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv", "matching_columns": [], "exact_matching_columns": 0}]