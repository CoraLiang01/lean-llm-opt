\Generated by Cardinal Operations

Maximize
   1142.6 x_('A',_'B')_06:40_f + 482.92 x_('A',_'B')_06:40_l 
   + 1133.9 x_('B',_'A')_20:25_f + 410.31 x_('B',_'A')_20:25_l 
   + 1123.69 x_('B',_'A')_09:05_f + 419.78 x_('B',_'A')_09:05_l 
   + 1284.5 x_('C',_'A')_08:15_f + 477.98 x_('C',_'A')_08:15_l 
   + 1285.73 x_('C',_'A')_16:55_f + 527.21 x_('C',_'A')_16:55_l 
   + 1134.68 x_('A',_'B')_22:45_f + 428.56 x_('A',_'B')_22:45_l 
   + 1138.14 x_('A',_'B')_19:10_f + 450.8 x_('A',_'B')_19:10_l 
   + 1285.82 x_('C',_'A')_07:40_f + 558.55 x_('C',_'A')_07:40_l 
   + 1292.44 x_('A',_'C')_09:45_f + 485.92 x_('A',_'C')_09:45_l 
   + 1138.59 x_('B',_'A')_18:50_f + 429.78 x_('B',_'A')_18:50_l 
   + 1269.77 x_('A',_'C')_23:00_f + 485.58 x_('A',_'C')_23:00_l 
Subject To
 capacity_constraint_('A',_'B')_06:40: 1.6 x_('A',_'B')_06:40_f 
   + x_('A',_'B')_06:40_l <= 187
 scale_constraint_('A',_'B')_06:40_f: 0.139137252 x_('A',_'B')_0 
   - x_('A',_'B')_06:40_f >= 0
 scale_constraint_('A',_'B')_06:40_l: 0.299098 x_('A',_'B')_0 
   - x_('A',_'B')_06:40_l >= 0
 capacity_constraint_('B',_'A')_20:25: 1.6 x_('B',_'A')_20:25_f 
   + x_('B',_'A')_20:25_l <= 187
 scale_constraint_('B',_'A')_20:25_f: 0.780656 x_('B',_'A')_0 
   - x_('B',_'A')_20:25_f >= 0
 scale_constraint_('B',_'A')_20:25_l: 1.332733 x_('B',_'A')_0 
   - x_('B',_'A')_20:25_l >= 0
 capacity_constraint_('B',_'A')_09:05: 1.6 x_('B',_'A')_09:05_f 
   + x_('B',_'A')_09:05_l <= 187
 scale_constraint_('B',_'A')_09:05_f: 1.043622 x_('B',_'A')_0 
   - x_('B',_'A')_09:05_f >= 0
 scale_constraint_('B',_'A')_09:05_l: 0.0625 x_('B',_'A')_0 
   - x_('B',_'A')_09:05_l >= 0
 capacity_constraint_('C',_'A')_08:15: 1.6 x_('C',_'A')_08:15_f 
   + x_('C',_'A')_08:15_l <= 187
 scale_constraint_('C',_'A')_08:15_f: 0.278102 x_('C',_'A')_0 
   - x_('C',_'A')_08:15_f >= 0
 scale_constraint_('C',_'A')_08:15_l: 0.108023 x_('C',_'A')_0 
   - x_('C',_'A')_08:15_l >= 0
 capacity_constraint_('C',_'A')_16:55: 1.6 x_('C',_'A')_16:55_f 
   + x_('C',_'A')_16:55_l <= 187
 scale_constraint_('C',_'A')_16:55_f: 0.089563 x_('C',_'A')_0 
   - x_('C',_'A')_16:55_f >= 0
 scale_constraint_('C',_'A')_16:55_l: 0.141624 x_('C',_'A')_0 
   - x_('C',_'A')_16:55_l >= 0
 capacity_constraint_('A',_'B')_22:45: 1.6 x_('A',_'B')_22:45_f 
   + x_('A',_'B')_22:45_l <= 187
 scale_constraint_('A',_'B')_22:45_f: 0.139137252 x_('A',_'B')_0 
   - x_('A',_'B')_22:45_f >= 0
 scale_constraint_('A',_'B')_22:45_l: 0.299098 x_('A',_'B')_0 
   - x_('A',_'B')_22:45_l >= 0
 capacity_constraint_('A',_'B')_19:10: 1.6 x_('A',_'B')_19:10_f 
   + x_('A',_'B')_19:10_l <= 187
 scale_constraint_('A',_'B')_19:10_f: 0.193098 x_('A',_'B')_0 
   - x_('A',_'B')_19:10_f >= 0
 scale_constraint_('A',_'B')_19:10_l: 0.112429 x_('A',_'B')_0 
   - x_('A',_'B')_19:10_l >= 0
 capacity_constraint_('C',_'A')_07:40: 1.6 x_('C',_'A')_07:40_f 
   + x_('C',_'A')_07:40_l <= 187
 scale_constraint_('C',_'A')_07:40_f: 0.228917 x_('C',_'A')_0 
   - x_('C',_'A')_07:40_f >= 0
 scale_constraint_('C',_'A')_07:40_l: 0.54522 x_('C',_'A')_0 
   - x_('C',_'A')_07:40_l >= 0
 capacity_constraint_('A',_'C')_09:45: 1.6 x_('A',_'C')_09:45_f 
   + x_('A',_'C')_09:45_l <= 187
 scale_constraint_('A',_'C')_09:45_f: 0.705471 x_('A',_'C')_0 
   - x_('A',_'C')_09:45_f >= 0
 scale_constraint_('A',_'C')_09:45_l: 0.0625 x_('A',_'C')_0 
   - x_('A',_'C')_09:45_l >= 0
 capacity_constraint_('B',_'A')_18:50: 1.6 x_('B',_'A')_18:50_f 
   + x_('B',_'A')_18:50_l <= 187
 scale_constraint_('B',_'A')_18:50_f: 0.780656 x_('B',_'A')_0 
   - x_('B',_'A')_18:50_f >= 0
 scale_constraint_('B',_'A')_18:50_l: 1.332733 x_('B',_'A')_0 
   - x_('B',_'A')_18:50_l >= 0
 capacity_constraint_('A',_'C')_23:00: 1.6 x_('A',_'C')_23:00_f 
   + x_('A',_'C')_23:00_l <= 187
 scale_constraint_('A',_'C')_23:00_f: 1.276491 x_('A',_'C')_0 
   - x_('A',_'C')_23:00_f >= 0
 scale_constraint_('A',_'C')_23:00_l: 0.0625 x_('A',_'C')_0 
   - x_('A',_'C')_23:00_l >= 0
 balance_constraint_('A',_'C'): x_('A',_'C')_09:45_f + x_('A',_'C')_09:45_l 
   + 0.87 x_('A',_'C')_23:00_f + x_('A',_'C')_23:00_l + 1.55 x_('A',_'C')_0 
   = 4812.5
 balance_constraint_('C',_'A'): 0.82 x_('C',_'A')_08:15_f 
   + 0.89 x_('C',_'A')_08:15_l + 0.7 x_('C',_'A')_16:55_f 
   + 0.48 x_('C',_'A')_16:55_l + x_('C',_'A')_07:40_f 
   + 0.97 x_('C',_'A')_07:40_l + 1.4 x_('C',_'A')_0 = 4807.43
 balance_constraint_('A',_'B'): 1.41 x_('A',_'B')_0 
   + 0.65 x_('A',_'B')_06:40_l + x_('A',_'B')_19:10_l 
   + 0.65 x_('A',_'B')_22:45_l + 0.74 x_('A',_'B')_19:10_f = 38965.86
 balance_constraint_('B',_'A'): 0.74 x_('B',_'A')_20:25_f 
   + 0.97 x_('B',_'A')_20:25_l + x_('B',_'A')_09:05_f 
   + 0.29 x_('B',_'A')_09:05_l + 0.74 x_('B',_'A')_18:50_f 
   + 0.97 x_('B',_'A')_18:50_l + 1.53 x_('B',_'A')_0 = 33210.71
END

