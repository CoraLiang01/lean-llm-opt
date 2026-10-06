Sets:
- Let S = {S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11} be the set of Walmart stores.
- Let C = {C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12} be the set of customer groups.

Parameters:
- demand_c: demand for customer group c ∈ C, from customer_demand.csv:
    demand_C1 = 11
    demand_C2 = 1148
    demand_C3 = 54
    demand_C4 = 833
    demand_C5 = 154
    demand_C6 = 551
    demand_C7 = 7081
    demand_C8 = 76
    demand_C9 = 66
    demand_C10 = 174
    demand_C11 = 15
    demand_C12 = 680

- supply_s: supply capacity for store s ∈ S, from supply_capacity.csv:
    supply_S1 = 4
    supply_S2 = 575
    supply_S3 = 1504
    supply_S4 = 178
    supply_S5 = 228
    supply_S6 = 50
    supply_S7 = 3
    supply_S8 = 6148
    supply_S9 = 6
    supply_S10 = 10673
    supply_S11 = 174

- cost_{s,c}: transportation cost per unit from store s to customer group c, from transportation_costs.csv (source order preserved):

    For S1:
        cost_{S1,C1} = 0.639144476970582
        cost_{S1,C2} = 49.71842803015729
        cost_{S1,C3} = 33.75857739960576
        cost_{S1,C4} = 1570.673110465785
        cost_{S1,C5} = 1370.4095474322417
        cost_{S1,C6} = 57.35307774277479
        cost_{S1,C7} = 57.18299486453194
        cost_{S1,C8} = 54.9209612366192
        cost_{S1,C9} = 1143.680909226399
        cost_{S1,C10} = 52.49127007738756
        cost_{S1,C11} = 606.4434399601076
        cost_{S1,C12} = 1192.4686514332489

    For S2:
        cost_{S2,C1} = 605.4786373569875
        cost_{S2,C2} = 64.53562572761275
        cost_{S2,C3} = 478.4779031378926
        cost_{S2,C4} = 887.0480739088434
        cost_{S2,C5} = 65.46111249492031
        cost_{S2,C6} = 71.93605217833378
        cost_{S2,C7} = 41.29015388498019
        cost_{S2,C8} = 70.36038207491039
        cost_{S2,C9} = 35.35892996332259
        cost_{S2,C10} = 1472.7481944140839
        cost_{S2,C11} = 0.6004591535232997
        cost_{S2,C12} = 49.86854015671519

    For S3:
        cost_{S3,C1} = 1139.0440074582496
        cost_{S3,C2} = 4.785056325458736
        cost_{S3,C3} = 1805.6214229758102
        cost_{S3,C4} = 1302.8958147418275
        cost_{S3,C5} = 2437.321229159901
        cost_{S3,C6} = 103.80368582531935
        cost_{S3,C7} = 774.6558236505713
        cost_{S3,C8} = 4.515988277174664
        cost_{S3,C9} = 879.7048537066717
        cost_{S3,C10} = 162.70556734409897
        cost_{S3,C11} = 1208.613484750161
        cost_{S3,C12} = 110.18688517926226

    For S4:
        cost_{S4,C1} = 69.26989890601938
        cost_{S4,C2} = 2105.485387219297
        cost_{S4,C3} = 869.6820232492624
        cost_{S4,C4} = 1494.8985656180187
        cost_{S4,C5} = 310.5376623181487
        cost_{S4,C6} = 98.15455717980421
        cost_{S4,C7} = 103.36918486373995
        cost_{S4,C8} = 1758.8783768888936
        cost_{S4,C9} = 97.28540713798621
        cost_{S4,C10} = 94.6504089308906
        cost_{S4,C11} = 1277.251451477508
        cost_{S4,C12} = 21.636190287574664

    For S5:
        cost_{S5,C1} = 980.4114260089906
        cost_{S5,C2} = 899.3108831856057
        cost_{S5,C3} = 1183.032552702089
        cost_{S5,C4} = 402.0986161964097
        cost_{S5,C5} = 81.78864123893297
        cost_{S5,C6} = 1115.6819455776936
        cost_{S5,C7} = 123.80427864011308
        cost_{S5,C8} = 1121.14687497079
        cost_{S5,C9} = 0.0024451264081394235
        cost_{S5,C10} = 1009.6451734576028
        cost_{S5,C11} = 35.348018297991366
        cost_{S5,C12} = 1625.434626662855

    For S6:
        cost_{S6,C1} = 1246.782499848912
        cost_{S6,C2} = 2105.7966672265507
        cost_{S6,C3} = 1014.3393213554372
        cost_{S6,C4} = 1494.6681439414933
        cost_{S6,C5} = 362.0173906933171
        cost_{S6,C6} = 98.17142420905459
        cost_{S6,C7} = 2170.405867933639
        cost_{S6,C8} = 97.7318771093979
        cost_{S6,C9} = 97.26834016110725
        cost_{S6,C10} = 1987.9907846635351
        cost_{S6,C11} = 70.94396914331519
        cost_{S6,C12} = 389.15980447937727

    For S7:
        cost_{S7,C1} = 57.1086015288379
        cost_{S7,C2} = 23.836168859030245
        cost_{S7,C3} = 78.10572975165614
        cost_{S7,C4} = 742.8068113300407
        cost_{S7,C5} = 1926.0796823736941
        cost_{S7,C6} = 454.3789956981779
        cost_{S7,C7} = 458.2901436941235
        cost_{S7,C8} = 465.9307664524444
        cost_{S7,C9} = 28.138607069878855
        cost_{S7,C10} = 524.6154260270081
        cost_{S7,C11} = 997.531783848061
        cost_{S7,C12} = 104.47794493215576

    For S8:
        cost_{S8,C1} = 981.2908605082814
        cost_{S8,C2} = 120.90130000942015
        cost_{S8,C3} = 1625.8206931791087
        cost_{S8,C4} = 1267.8229294135008
        cost_{S8,C5} = 2569.6446053909003
        cost_{S8,C6} = 13.471811837256363
        cost_{S8,C7} = 815.1525428026629
        cost_{S8,C8} = 253.42349641458793
        cost_{S8,C9} = 43.76562945456531
        cost_{S8,C10} = 275.9784134803488
        cost_{S8,C11} = 1228.06989342366
        cost_{S8,C12} = 103.48323020673556

    For S9:
        cost_{S9,C1} = 30.532779511898102
        cost_{S9,C2} = 1444.8594995969975
        cost_{S9,C3} = 173.5547323639261
        cost_{S9,C4} = 1307.3913121142912
        cost_{S9,C5} = 965.2012304156898
        cost_{S9,C6} = 1843.7769498110006
        cost_{S9,C7} = 1483.6408846054035
        cost_{S9,C8} = 85.32209952688736
        cost_{S9,C9} = 1353.500934450796
        cost_{S9,C10} = 1485.9153764236357
        cost_{S9,C11} = 29.423790844675874
        cost_{S9,C12} = 26.619419605630917

    For S10:
        cost_{S10,C1} = 94.11093956131819
        cost_{S10,C2} = 1422.9971302244805
        cost_{S10,C3} = 1470.776907673336
        cost_{S10,C4} = 1419.3382251704456
        cost_{S10,C5} = 38.94527784177093
        cost_{S10,C6} = 72.20112949102915
        cost_{S10,C7} = 2040.4605902793303
        cost_{S10,C8} = 1542.702557551204
        cost_{S10,C9} = 1803.8001691896025
        cost_{S10,C10} = 72.94365832842001
        cost_{S10,C11} = 2181.454205937846
        cost_{S10,C12} = 973.5515553238279

    For S11:
        cost_{S11,C1} = 1032.9073835595004
        cost_{S11,C2} = 166.30184787458444
        cost_{S11,C3} = 1620.4767053028727
        cost_{S11,C4} = 64.668341345234
        cost_{S11,C5} = 2000.50917314264
        cost_{S11,C6} = 0.0028957952749427158
        cost_{S11,C7} = 47.038371401381845
        cost_{S11,C8} = 52.99221132466169
        cost_{S11,C9} = 1115.6336172632205
        cost_{S11,C10} = 129.79338189093912
        cost_{S11,C11} = 1295.09784482215
        cost_{S11,C12} = 2330.76820970791

Decision Variables:
- x_{s,c} ≥ 0: quantity transported from store s ∈ S to customer group c ∈ C (continuous, non-negative).

Objective:
Minimize total transportation cost:
minimize
∑_{s∈S} ∑_{c∈C} cost_{s,c} * x_{s,c}

Constraints:
1. Demand satisfaction for each customer group:
   For all c ∈ C:
   ∑_{s∈S} x_{s,c} = demand_c

2. Supply capacity for each store:
   For all s ∈ S:
   ∑_{c∈C} x_{s,c} ≤ supply_s

3. Non-negativity:
   For all s ∈ S, c ∈ C:
   x_{s,c} ≥ 0

Full Model:
minimize
∑_{s∈S} ∑_{c∈C} cost_{s,c} * x_{s,c}

subject to
∀ c ∈ C:  ∑_{s∈S} x_{s,c} = demand_c
∀ s ∈ S:  ∑_{c∈C} x_{s,c} ≤ supply_s
∀ s ∈ S, c ∈ C:  x_{s,c} ≥ 0

All identifiers and coefficients are as listed above, preserving source order.