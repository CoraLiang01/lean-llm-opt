Maximize
    obj: 0.75 x_A1_I + 1.15 x_A1_II + 0.7753 x_A2_I + 1.3611 x_A2_II + 1.9148 x_A2_III - 0.375 x_B1_I - 0.5 x_B1_II - 0.4474285714285714285714285716 x_B2_I - 1.230428571428571428571428572 x_B2_III - 0.35 x_B3_I
Subject To
    capacity_A1: 5 x_A1_I + 10 x_A1_II <= 6000
    capacity_A2: 7 x_A2_I + 9 x_A2_II + 12 x_A2_III <= 10000
    capacity_B1: 6 x_B1_I + 8 x_B1_II <= 4000
    capacity_B2: 4 x_B2_I + 11 x_B2_III <= 7000
    capacity_B3: 7 x_B3_I <= 4000
    flow_I: x_A1_I + x_A2_I - x_B1_I - x_B2_I - x_B3_I = 0
    flow_II: x_A1_II + x_A2_II - x_B1_II = 0
    flow_III: x_A2_III - x_B2_III = 0
End
