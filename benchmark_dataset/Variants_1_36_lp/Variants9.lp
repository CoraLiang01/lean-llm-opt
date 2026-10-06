Minimize
    obj: y_P1 + y_P2 + y_P3 + y_P4 + y_P5 + y_P6 + y_P7 + y_P8 + y_P9
    Subject To
    demand_A: 4 y_P1 + 2 y_P5 + y_P6 + y_P8 + 2 y_P9 >= 24
    demand_B: 3 y_P2 + y_P5 + y_P7 + y_P8 >= 18
    demand_C: 2 y_P3 + y_P6 + y_P8 >= 12
    demand_D: 2 y_P4 + y_P7 + y_P9 >= 10
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
    Generals
    y_P1 y_P2 y_P3 y_P4 y_P5 y_P6 y_P7 y_P8 y_P9
    End
