\Generated by Cardinal Operations

Maximize
   1131.33 x_('B',_'A')_06:25_f + 424.19 x_('B',_'A')_06:25_l 
   + 1142.6 x_('A',_'B')_06:40_f + 482.92 x_('A',_'B')_06:40_l 
   + 1285.82 x_('C',_'A')_07:40_f + 558.55 x_('C',_'A')_07:40_l 
   + 1138.19 x_('A',_'B')_07:55_f + 441.83 x_('A',_'B')_07:55_l 
   + 1284.5 x_('C',_'A')_08:15_f + 477.98 x_('C',_'A')_08:15_l 
   + 1123.69 x_('B',_'A')_09:05_f + 419.78 x_('B',_'A')_09:05_l 
   + 1292.44 x_('A',_'C')_09:45_f + 485.92 x_('A',_'C')_09:45_l 
   + 1140.3 x_('A',_'B')_11:20_f + 429.26 x_('A',_'B')_11:20_l 
   + 1139.8 x_('B',_'A')_15:40_f + 441.86 x_('B',_'A')_15:40_l 
   + 1285.73 x_('C',_'A')_16:55_f + 527.21 x_('C',_'A')_16:55_l 
   + 1145.54 x_('A',_'B')_17:05_f + 498.89 x_('A',_'B')_17:05_l 
   + 1286.29 x_('A',_'C')_17:25_f + 540.1900000000001 x_('A',_'C')_17:25_l 
   + 1274.28 x_('C',_'A')_18:30_f + 474.82 x_('C',_'A')_18:30_l 
   + 1138.59 x_('B',_'A')_18:50_f + 429.78 x_('B',_'A')_18:50_l 
   + 1285.64 x_('A',_'C')_19:05_f + 499.36 x_('A',_'C')_19:05_l 
   + 1138.14 x_('A',_'B')_19:10_f + 450.8 x_('A',_'B')_19:10_l 
   + 1133.9 x_('B',_'A')_20:25_f + 410.31 x_('B',_'A')_20:25_l 
   + 1134.68 x_('A',_'B')_22:45_f + 428.56 x_('A',_'B')_22:45_l 
   + 1269.77 x_('A',_'C')_23:00_f + 485.58 x_('A',_'C')_23:00_l 
Subject To
 capacity_constraint_('B',_'A')_06:25: 1.5 x_('B',_'A')_06:25_f 
   + x_('B',_'A')_06:25_l <= 187
 scale_constraint_('B',_'A')_06:25_f: 0.186922 x_('B',_'A')_0 
   - x_('B',_'A')_06:25_f >= 0
 scale_constraint_('B',_'A')_06:25_l: 0.0625 x_('B',_'A')_0 
   - x_('B',_'A')_06:25_l >= 0
 M_constraint_('B',_'A')_f: - 10000000 y_('B',_'A')_06:25 
   + x_('B',_'A')_06:25_f <= 0
 M_constraint_('B',_'A')_l: - 10000000 y_('B',_'A')_06:25 
   + x_('B',_'A')_06:25_l <= 0
 capacity_constraint_('A',_'B')_06:40: 1.5 x_('A',_'B')_06:40_f 
   + x_('A',_'B')_06:40_l <= 187
 scale_constraint_('A',_'B')_06:40_f: 0.139137252 x_('A',_'B')_0 
   - x_('A',_'B')_06:40_f >= 0
 scale_constraint_('A',_'B')_06:40_l: 0.299098 x_('A',_'B')_0 
   - x_('A',_'B')_06:40_l >= 0
 M_constraint_('A',_'B')_f: - 10000000 y_('A',_'B')_06:40 
   + x_('A',_'B')_06:40_f <= 0
 M_constraint_('A',_'B')_l: - 10000000 y_('A',_'B')_06:40 
   + x_('A',_'B')_06:40_l <= 0
 capacity_constraint_('C',_'A')_07:40: 1.5 x_('C',_'A')_07:40_f 
   + x_('C',_'A')_07:40_l <= 187
 scale_constraint_('C',_'A')_07:40_f: 0.228917 x_('C',_'A')_0 
   - x_('C',_'A')_07:40_f >= 0
 scale_constraint_('C',_'A')_07:40_l: 0.54522 x_('C',_'A')_0 
   - x_('C',_'A')_07:40_l >= 0
 M_constraint_('C',_'A')_f: - 10000000 y_('C',_'A')_07:40 
   + x_('C',_'A')_07:40_f <= 0
 M_constraint_('C',_'A')_l: - 10000000 y_('C',_'A')_07:40 
   + x_('C',_'A')_07:40_l <= 0
 capacity_constraint_('A',_'B')_07:55: 1.5 x_('A',_'B')_07:55_f 
   + x_('A',_'B')_07:55_l <= 187
 scale_constraint_('A',_'B')_07:55_f: 0.139137252 x_('A',_'B')_0 
   - x_('A',_'B')_07:55_f >= 0
 scale_constraint_('A',_'B')_07:55_l: 0.299098 x_('A',_'B')_0 
   - x_('A',_'B')_07:55_l >= 0
 - 10000000 y_('A',_'B')_07:55 + x_('A',_'B')_07:55_f <= 0
 - 10000000 y_('A',_'B')_07:55 + x_('A',_'B')_07:55_l <= 0
 capacity_constraint_('C',_'A')_08:15: 1.5 x_('C',_'A')_08:15_f 
   + x_('C',_'A')_08:15_l <= 187
 scale_constraint_('C',_'A')_08:15_f: 0.278102 x_('C',_'A')_0 
   - x_('C',_'A')_08:15_f >= 0
 scale_constraint_('C',_'A')_08:15_l: 0.108023 x_('C',_'A')_0 
   - x_('C',_'A')_08:15_l >= 0
 - 10000000 y_('C',_'A')_08:15 + x_('C',_'A')_08:15_f <= 0
 - 10000000 y_('C',_'A')_08:15 + x_('C',_'A')_08:15_l <= 0
 capacity_constraint_('B',_'A')_09:05: 1.5 x_('B',_'A')_09:05_f 
   + x_('B',_'A')_09:05_l <= 187
 scale_constraint_('B',_'A')_09:05_f: 1.043622 x_('B',_'A')_0 
   - x_('B',_'A')_09:05_f >= 0
 scale_constraint_('B',_'A')_09:05_l: 0.0625 x_('B',_'A')_0 
   - x_('B',_'A')_09:05_l >= 0
 - 10000000 y_('B',_'A')_09:05 + x_('B',_'A')_09:05_f <= 0
 - 10000000 y_('B',_'A')_09:05 + x_('B',_'A')_09:05_l <= 0
 capacity_constraint_('A',_'C')_09:45: 1.5 x_('A',_'C')_09:45_f 
   + x_('A',_'C')_09:45_l <= 187
 scale_constraint_('A',_'C')_09:45_f: 0.705471 x_('A',_'C')_0 
   - x_('A',_'C')_09:45_f >= 0
 scale_constraint_('A',_'C')_09:45_l: 0.0625 x_('A',_'C')_0 
   - x_('A',_'C')_09:45_l >= 0
 M_constraint_('A',_'C')_f: - 10000000 y_('A',_'C')_09:45 
   + x_('A',_'C')_09:45_f <= 0
 M_constraint_('A',_'C')_l: - 10000000 y_('A',_'C')_09:45 
   + x_('A',_'C')_09:45_l <= 0
 capacity_constraint_('A',_'B')_11:20: 1.5 x_('A',_'B')_11:20_f 
   + x_('A',_'B')_11:20_l <= 187
 scale_constraint_('A',_'B')_11:20_f: 0.342687 x_('A',_'B')_0 
   - x_('A',_'B')_11:20_f >= 0
 scale_constraint_('A',_'B')_11:20_l: 0.24677 x_('A',_'B')_0 
   - x_('A',_'B')_11:20_l >= 0
 - 10000000 y_('A',_'B')_11:20 + x_('A',_'B')_11:20_f <= 0
 - 10000000 y_('A',_'B')_11:20 + x_('A',_'B')_11:20_l <= 0
 capacity_constraint_('B',_'A')_15:40: 1.5 x_('B',_'A')_15:40_f 
   + x_('B',_'A')_15:40_l <= 187
 scale_constraint_('B',_'A')_15:40_f: 0.177309 x_('B',_'A')_0 
   - x_('B',_'A')_15:40_f >= 0
 scale_constraint_('B',_'A')_15:40_l: 0.471942 x_('B',_'A')_0 
   - x_('B',_'A')_15:40_l >= 0
 - 10000000 y_('B',_'A')_15:40 + x_('B',_'A')_15:40_f <= 0
 - 10000000 y_('B',_'A')_15:40 + x_('B',_'A')_15:40_l <= 0
 capacity_constraint_('C',_'A')_16:55: 1.5 x_('C',_'A')_16:55_f 
   + x_('C',_'A')_16:55_l <= 187
 scale_constraint_('C',_'A')_16:55_f: 0.089563 x_('C',_'A')_0 
   - x_('C',_'A')_16:55_f >= 0
 scale_constraint_('C',_'A')_16:55_l: 0.141624 x_('C',_'A')_0 
   - x_('C',_'A')_16:55_l >= 0
 - 10000000 y_('C',_'A')_16:55 + x_('C',_'A')_16:55_f <= 0
 - 10000000 y_('C',_'A')_16:55 + x_('C',_'A')_16:55_l <= 0
 capacity_constraint_('A',_'B')_17:05: 1.5 x_('A',_'B')_17:05_f 
   + x_('A',_'B')_17:05_l <= 187
 scale_constraint_('A',_'B')_17:05_f: 0.06374299999999999 x_('A',_'B')_0 
   - x_('A',_'B')_17:05_f >= 0
 scale_constraint_('A',_'B')_17:05_l: 0.041724 x_('A',_'B')_0 
   - x_('A',_'B')_17:05_l >= 0
 - 10000000 y_('A',_'B')_17:05 + x_('A',_'B')_17:05_f <= 0
 - 10000000 y_('A',_'B')_17:05 + x_('A',_'B')_17:05_l <= 0
 capacity_constraint_('A',_'C')_17:25: 1.5 x_('A',_'C')_17:25_f 
   + x_('A',_'C')_17:25_l <= 187
 scale_constraint_('A',_'C')_17:25_f: 0.395947 x_('A',_'C')_0 
   - x_('A',_'C')_17:25_f >= 0
 scale_constraint_('A',_'C')_17:25_l: 0.0625 x_('A',_'C')_0 
   - x_('A',_'C')_17:25_l >= 0
 - 10000000 y_('A',_'C')_17:25 + x_('A',_'C')_17:25_f <= 0
 - 10000000 y_('A',_'C')_17:25 + x_('A',_'C')_17:25_l <= 0
 capacity_constraint_('C',_'A')_18:30: 1.5 x_('C',_'A')_18:30_f 
   + x_('C',_'A')_18:30_l <= 187
 scale_constraint_('C',_'A')_18:30_f: 0.475475 x_('C',_'A')_0 
   - x_('C',_'A')_18:30_f >= 0
 scale_constraint_('C',_'A')_18:30_l: 0.647628 x_('C',_'A')_0 
   - x_('C',_'A')_18:30_l >= 0
 - 10000000 y_('C',_'A')_18:30 + x_('C',_'A')_18:30_f <= 0
 - 10000000 y_('C',_'A')_18:30 + x_('C',_'A')_18:30_l <= 0
 capacity_constraint_('B',_'A')_18:50: 1.5 x_('B',_'A')_18:50_f 
   + x_('B',_'A')_18:50_l <= 187
 scale_constraint_('B',_'A')_18:50_f: 0.780656 x_('B',_'A')_0 
   - x_('B',_'A')_18:50_f >= 0
 scale_constraint_('B',_'A')_18:50_l: 1.332733 x_('B',_'A')_0 
   - x_('B',_'A')_18:50_l >= 0
 - 10000000 y_('B',_'A')_18:50 + x_('B',_'A')_18:50_f <= 0
 - 10000000 y_('B',_'A')_18:50 + x_('B',_'A')_18:50_l <= 0
 capacity_constraint_('A',_'C')_19:05: 1.5 x_('A',_'C')_19:05_f 
   + x_('A',_'C')_19:05_l <= 187
 scale_constraint_('A',_'C')_19:05_f: 1.548071 x_('A',_'C')_0 
   - x_('A',_'C')_19:05_f >= 0
 scale_constraint_('A',_'C')_19:05_l: 0.0625 x_('A',_'C')_0 
   - x_('A',_'C')_19:05_l >= 0
 - 10000000 y_('A',_'C')_19:05 + x_('A',_'C')_19:05_f <= 0
 - 10000000 y_('A',_'C')_19:05 + x_('A',_'C')_19:05_l <= 0
 capacity_constraint_('A',_'B')_19:10: 1.5 x_('A',_'B')_19:10_f 
   + x_('A',_'B')_19:10_l <= 187
 scale_constraint_('A',_'B')_19:10_f: 0.193098 x_('A',_'B')_0 
   - x_('A',_'B')_19:10_f >= 0
 scale_constraint_('A',_'B')_19:10_l: 0.112429 x_('A',_'B')_0 
   - x_('A',_'B')_19:10_l >= 0
 - 10000000 y_('A',_'B')_19:10 + x_('A',_'B')_19:10_f <= 0
 - 10000000 y_('A',_'B')_19:10 + x_('A',_'B')_19:10_l <= 0
 capacity_constraint_('B',_'A')_20:25: 1.5 x_('B',_'A')_20:25_f 
   + x_('B',_'A')_20:25_l <= 187
 scale_constraint_('B',_'A')_20:25_f: 0.780656 x_('B',_'A')_0 
   - x_('B',_'A')_20:25_f >= 0
 scale_constraint_('B',_'A')_20:25_l: 1.332733 x_('B',_'A')_0 
   - x_('B',_'A')_20:25_l >= 0
 - 10000000 y_('B',_'A')_20:25 + x_('B',_'A')_20:25_f <= 0
 - 10000000 y_('B',_'A')_20:25 + x_('B',_'A')_20:25_l <= 0
 capacity_constraint_('A',_'B')_22:45: 1.5 x_('A',_'B')_22:45_f 
   + x_('A',_'B')_22:45_l <= 187
 scale_constraint_('A',_'B')_22:45_f: 0.139137252 x_('A',_'B')_0 
   - x_('A',_'B')_22:45_f >= 0
 scale_constraint_('A',_'B')_22:45_l: 0.299098 x_('A',_'B')_0 
   - x_('A',_'B')_22:45_l >= 0
 - 10000000 y_('A',_'B')_22:45 + x_('A',_'B')_22:45_f <= 0
 - 10000000 y_('A',_'B')_22:45 + x_('A',_'B')_22:45_l <= 0
 capacity_constraint_('A',_'C')_23:00: 1.5 x_('A',_'C')_23:00_f 
   + x_('A',_'C')_23:00_l <= 187
 scale_constraint_('A',_'C')_23:00_f: 1.276491 x_('A',_'C')_0 
   - x_('A',_'C')_23:00_f >= 0
 scale_constraint_('A',_'C')_23:00_l: 0.0625 x_('A',_'C')_0 
   - x_('A',_'C')_23:00_l >= 0
 - 10000000 y_('A',_'C')_23:00 + x_('A',_'C')_23:00_f <= 0
 - 10000000 y_('A',_'C')_23:00 + x_('A',_'C')_23:00_l <= 0
 balance_constraint_('B',_'A'): 0.1 x_('B',_'A')_06:25_f 
   + 0.29 x_('B',_'A')_06:25_l + x_('B',_'A')_09:05_f 
   + 0.29 x_('B',_'A')_09:05_l + x_('B',_'A')_15:40_f + 1.53 x_('B',_'A')_0 
   + 0.74 x_('B',_'A')_18:50_f + 0.97 x_('B',_'A')_18:50_l 
   + 0.74 x_('B',_'A')_20:25_f + 0.97 x_('B',_'A')_20:25_l = 33210.71
 balance_constraint_('A',_'B'): 1.41 x_('A',_'B')_0 
   + 0.65 x_('A',_'B')_06:40_l + 0.65 x_('A',_'B')_22:45_l 
   + 0.65 x_('A',_'B')_07:55_l + 0.67 x_('A',_'B')_11:20_f 
   + x_('A',_'B')_11:20_l + x_('A',_'B')_17:05_f + x_('A',_'B')_17:05_l 
   + 0.74 x_('A',_'B')_19:10_f + x_('A',_'B')_19:10_l = 38965.86
 balance_constraint_('C',_'A'): x_('C',_'A')_07:40_f 
   + 0.97 x_('C',_'A')_07:40_l + 0.82 x_('C',_'A')_08:15_f 
   + 0.89 x_('C',_'A')_08:15_l + 0.7 x_('C',_'A')_16:55_f 
   + 0.48 x_('C',_'A')_16:55_l + 0.74 x_('C',_'A')_18:30_f 
   + 0.85 x_('C',_'A')_18:30_l + 1.4 x_('C',_'A')_0 = 4807.43
 balance_constraint_('A',_'C'): x_('A',_'C')_09:45_f + x_('A',_'C')_09:45_l 
   + 0.5 x_('A',_'C')_17:25_f + x_('A',_'C')_17:25_l 
   + 0.88 x_('A',_'C')_19:05_f + x_('A',_'C')_19:05_l 
   + 0.87 x_('A',_'C')_23:00_f + x_('A',_'C')_23:00_l + 1.55 x_('A',_'C')_0 
   = 4812.5
 Flow_Conservation_constraint_A: y_('A',_'B')_06:40 + y_('A',_'B')_07:55 
   + y_('A',_'B')_11:20 + y_('A',_'B')_17:05 + y_('A',_'B')_19:10 
   + y_('A',_'B')_22:45 + y_('A',_'C')_09:45 + y_('A',_'C')_17:25 
   + y_('A',_'C')_19:05 + y_('A',_'C')_23:00 - y_('B',_'A')_06:25 
   - y_('B',_'A')_09:05 - y_('B',_'A')_15:40 - y_('B',_'A')_18:50 
   - y_('B',_'A')_20:25 - y_('C',_'A')_07:40 - y_('C',_'A')_08:15 
   - y_('C',_'A')_16:55 - y_('C',_'A')_18:30 = 0
 Flow_Conservation_constraint_B: y_('B',_'A')_06:25 + y_('B',_'A')_09:05 
   + y_('B',_'A')_15:40 + y_('B',_'A')_18:50 + y_('B',_'A')_20:25 
   - y_('A',_'B')_06:40 - y_('A',_'B')_07:55 - y_('A',_'B')_11:20 
   - y_('A',_'B')_17:05 - y_('A',_'B')_19:10 - y_('A',_'B')_22:45 = 0
 Flow_Conservation_constraint_C: y_('C',_'A')_07:40 + y_('C',_'A')_08:15 
   + y_('C',_'A')_16:55 + y_('C',_'A')_18:30 - y_('A',_'C')_09:45 
   - y_('A',_'C')_17:25 - y_('A',_'C')_19:05 - y_('A',_'C')_23:00 = 0
 cardinality_constraint: y_('B',_'A')_06:25 + y_('B',_'A')_09:05 
   + y_('B',_'A')_15:40 + y_('B',_'A')_18:50 + y_('B',_'A')_20:25 
   + y_('A',_'B')_06:40 + y_('A',_'B')_07:55 + y_('A',_'B')_11:20 
   + y_('A',_'B')_17:05 + y_('A',_'B')_19:10 + y_('A',_'B')_22:45 
   + y_('C',_'A')_07:40 + y_('C',_'A')_08:15 + y_('C',_'A')_16:55 
   + y_('C',_'A')_18:30 + y_('A',_'C')_09:45 + y_('A',_'C')_17:25 
   + y_('A',_'C')_19:05 + y_('A',_'C')_23:00 <= 4
Binaries
 y_('B',_'A')_06:25  y_('A',_'B')_06:40  y_('C',_'A')_07:40  
 y_('A',_'B')_07:55  y_('C',_'A')_08:15  y_('B',_'A')_09:05  
 y_('A',_'C')_09:45  y_('A',_'B')_11:20  y_('B',_'A')_15:40  
 y_('C',_'A')_16:55  y_('A',_'B')_17:05  y_('A',_'C')_17:25  
 y_('C',_'A')_18:30  y_('B',_'A')_18:50  y_('A',_'C')_19:05  
 y_('A',_'B')_19:10  y_('B',_'A')_20:25  y_('A',_'B')_22:45  
 y_('A',_'C')_23:00  
END

