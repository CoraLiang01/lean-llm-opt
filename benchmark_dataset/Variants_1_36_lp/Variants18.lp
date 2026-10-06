Minimize
    obj: y_P1 + y_P2 + y_P3 + y_P4 + y_P5 + y_P6 + y_P7 + y_P8 + y_P9 + y_P10
    Subject To
    demand_A: 3 y_P1 + y_P5 + 2 y_P6 + y_P8 + y_P9 >= 18
    demand_B: 2 y_P2 + y_P5 + y_P7 + 2 y_P9 >= 14
    demand_C: y_P2 + 2 y_P3 + y_P6 + y_P7 + y_P10 >= 12
    demand_D: y_P3 + 2 y_P4 + y_P5 + y_P8 + y_P10 >= 10
    demand_E: y_P4 + y_P7 + y_P8 + y_P10 >= 8
    Bounds
    y_P1 >= 0
    y_P2 >= 0
    y_P3 >= 0
    y_P4 >= 0
    y_P5 >= 0
    y_P6 >= 0
    y_P7 >= 0
    y_P8 >= 0
    y_P9 >= 0
    y_P10 >= 0
    Generals
    y_P1 y_P2 y_P3 y_P4 y_P5 y_P6 y_P7 y_P8 y_P9 y_P10
    End
