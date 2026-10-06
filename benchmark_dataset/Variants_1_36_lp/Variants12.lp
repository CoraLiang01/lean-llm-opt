Minimize
    obj: 3 x_P1_R1 + 260 y_P1_R1 + 4 x_P1_R2 + 300 y_P1_R2 + 18 x_P1_R3 + 620 y_P1_R3 + 20 x_P1_R4
      + 670 y_P1_R4 + 22 x_P1_R5 + 710 y_P1_R5 + 19 x_P1_R6 + 660 y_P1_R6 + 17 x_P2_R1 + 610 y_P2_R1
      + 16 x_P2_R2 + 590 y_P2_R2 + 4 x_P2_R3 + 280 y_P2_R3 + 5 x_P2_R4 + 310 y_P2_R4 + 20 x_P2_R5
      + 690 y_P2_R5 + 18 x_P2_R6 + 640 y_P2_R6 + 21 x_P3_R1 + 700 y_P3_R1 + 19 x_P3_R2 + 650 y_P3_R2
      + 18 x_P3_R3 + 630 y_P3_R3 + 17 x_P3_R4 + 600 y_P3_R4 + 3 x_P3_R5 + 250 y_P3_R5 + 4 x_P3_R6
      + 290 y_P3_R6
    Subject To
    demand_R1: x_P1_R1 + x_P2_R1 + x_P3_R1 >= 70
    demand_R2: x_P1_R2 + x_P2_R2 + x_P3_R2 >= 85
    demand_R3: x_P1_R3 + x_P2_R3 + x_P3_R3 >= 75
    demand_R4: x_P1_R4 + x_P2_R4 + x_P3_R4 >= 65
    demand_R5: x_P1_R5 + x_P2_R5 + x_P3_R5 >= 95
    demand_R6: x_P1_R6 + x_P2_R6 + x_P3_R6 >= 80
    supply_P1: x_P1_R1 + x_P1_R2 + x_P1_R3 + x_P1_R4 + x_P1_R5 + x_P1_R6 <= 190
    supply_P2: x_P2_R1 + x_P2_R2 + x_P2_R3 + x_P2_R4 + x_P2_R5 + x_P2_R6 <= 160
    supply_P3: x_P3_R1 + x_P3_R2 + x_P3_R3 + x_P3_R4 + x_P3_R5 + x_P3_R6 <= 150
    route_link_P1_R1: x_P1_R1 - 70 y_P1_R1 <= 0
    route_link_P1_R2: x_P1_R2 - 85 y_P1_R2 <= 0
    route_link_P1_R3: x_P1_R3 - 75 y_P1_R3 <= 0
    route_link_P1_R4: x_P1_R4 - 65 y_P1_R4 <= 0
    route_link_P1_R5: x_P1_R5 - 95 y_P1_R5 <= 0
    route_link_P1_R6: x_P1_R6 - 80 y_P1_R6 <= 0
    route_link_P2_R1: x_P2_R1 - 70 y_P2_R1 <= 0
    route_link_P2_R2: x_P2_R2 - 85 y_P2_R2 <= 0
    route_link_P2_R3: x_P2_R3 - 75 y_P2_R3 <= 0
    route_link_P2_R4: x_P2_R4 - 65 y_P2_R4 <= 0
    route_link_P2_R5: x_P2_R5 - 95 y_P2_R5 <= 0
    route_link_P2_R6: x_P2_R6 - 80 y_P2_R6 <= 0
    route_link_P3_R1: x_P3_R1 - 70 y_P3_R1 <= 0
    route_link_P3_R2: x_P3_R2 - 85 y_P3_R2 <= 0
    route_link_P3_R3: x_P3_R3 - 75 y_P3_R3 <= 0
    route_link_P3_R4: x_P3_R4 - 65 y_P3_R4 <= 0
    route_link_P3_R5: x_P3_R5 - 95 y_P3_R5 <= 0
    route_link_P3_R6: x_P3_R6 - 80 y_P3_R6 <= 0
    Bounds
    x_P1_R1 >= 0
    x_P1_R2 >= 0
    x_P1_R3 >= 0
    x_P1_R4 >= 0
    x_P1_R5 >= 0
    x_P1_R6 >= 0
    x_P2_R1 >= 0
    x_P2_R2 >= 0
    x_P2_R3 >= 0
    x_P2_R4 >= 0
    x_P2_R5 >= 0
    x_P2_R6 >= 0
    x_P3_R1 >= 0
    x_P3_R2 >= 0
    x_P3_R3 >= 0
    x_P3_R4 >= 0
    x_P3_R5 >= 0
    x_P3_R6 >= 0
    Binaries
    y_P1_R1 y_P1_R2 y_P1_R3 y_P1_R4 y_P1_R5 y_P1_R6 y_P2_R1 y_P2_R2 y_P2_R3 y_P2_R4 y_P2_R5 y_P2_R6
    y_P3_R1 y_P3_R2 y_P3_R3 y_P3_R4 y_P3_R5 y_P3_R6
    End
