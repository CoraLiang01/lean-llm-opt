\ Model BusRouteStaffScheduling
    \ LP format - for model browsing. Use MPS format to capture full model detail.
    Minimize
    x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[11]
    + x[12] + x[13] + x[14] + x[15] + x[16] + x[17] + x[18] + x[19] + x[20] + x[21]
    + x[22] + x[23]
    Subject To
    Demand_Hour_00_Met: x[0] + x[21] + x[22] + x[23] >= 20
    Demand_Hour_01_Met: x[0] + x[1] + x[22] + x[23] >= 18
    Demand_Hour_02_Met: x[0] + x[1] + x[2] + x[23] >= 15
    Demand_Hour_03_Met: x[0] + x[1] + x[2] + x[3] >= 15
    Demand_Hour_04_Met: x[1] + x[2] + x[3] + x[4] >= 20
    Demand_Hour_05_Met: x[2] + x[3] + x[4] + x[5] >= 30
    Demand_Hour_06_Met: x[3] + x[4] + x[5] + x[6] >= 60
    Demand_Hour_07_Met: x[4] + x[5] + x[6] + x[7] >= 70
    Demand_Hour_08_Met: x[5] + x[6] + x[7] + x[8] >= 50
    Demand_Hour_09_Met: x[6] + x[7] + x[8] + x[9] >= 55
    Demand_Hour_10_Met: x[7] + x[8] + x[9] + x[10] >= 65
    Demand_Hour_11_Met: x[8] + x[9] + x[10] + x[11] >= 75
    Demand_Hour_12_Met: x[9] + x[10] + x[11] + x[12] >= 80
    Demand_Hour_13_Met: x[10] + x[11] + x[12] + x[13] >= 70
    Demand_Hour_14_Met: x[11] + x[12] + x[13] + x[14] >= 60
    Demand_Hour_15_Met: x[12] + x[13] + x[14] + x[15] >= 55
    Demand_Hour_16_Met: x[13] + x[14] + x[15] + x[16] >= 60
    Demand_Hour_17_Met: x[14] + x[15] + x[16] + x[17] >= 75
    Demand_Hour_18_Met: x[15] + x[16] + x[17] + x[18] >= 85
    Demand_Hour_19_Met: x[16] + x[17] + x[18] + x[19] >= 70
    Demand_Hour_20_Met: x[17] + x[18] + x[19] + x[20] >= 50
    Demand_Hour_21_Met: x[18] + x[19] + x[20] + x[21] >= 40
    Demand_Hour_22_Met: x[19] + x[20] + x[21] + x[22] >= 35
    Demand_Hour_23_Met: x[20] + x[21] + x[22] + x[23] >= 25
    Bounds
    Generals
    x[0] x[1] x[2] x[3] x[4] x[5] x[6] x[7] x[8] x[9] x[10] x[11] x[12] x[13] x[14] x[15]
    x[16] x[17] x[18] x[19] x[20] x[21] x[22] x[23]
    End
