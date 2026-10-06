Minimize
    obj: 2 f_S1_H1 + 6 f_S1_H2 + 4 f_S2_H1 + 3 f_S2_H2 + 7 f_S3_H1 + 2 f_S3_H2 + 3 f_H1_C1
      + 4 f_H1_C2 + 7 f_H1_C3 + 8 f_H1_C4 + 8 f_H2_C1 + 6 f_H2_C2 + 3 f_H2_C3 + 4 f_H2_C4
    Subject To
    source_supply_S1: f_S1_H1 + f_S1_H2 <= 120
    source_supply_S2: f_S2_H1 + f_S2_H2 <= 100
    source_supply_S3: f_S3_H1 + f_S3_H2 <= 90
    customer_demand_C1: f_H1_C1 + f_H2_C1 >= 70
    customer_demand_C2: f_H1_C2 + f_H2_C2 >= 80
    customer_demand_C3: f_H1_C3 + f_H2_C3 >= 60
    customer_demand_C4: f_H1_C4 + f_H2_C4 >= 90
    hub_balance_H1: f_S1_H1 + f_S2_H1 + f_S3_H1 - f_H1_C1 - f_H1_C2 - f_H1_C3 - f_H1_C4 = 0
    hub_balance_H2: f_S1_H2 + f_S2_H2 + f_S3_H2 - f_H2_C1 - f_H2_C2 - f_H2_C3 - f_H2_C4 = 0
    hub_throughput_H1: f_S1_H1 + f_S2_H1 + f_S3_H1 <= 170
    hub_throughput_H2: f_S1_H2 + f_S2_H2 + f_S3_H2 <= 160
    Bounds
    f_S1_H1 >= 0
    f_S1_H2 >= 0
    f_S2_H1 >= 0
    f_S2_H2 >= 0
    f_S3_H1 >= 0
    f_S3_H2 >= 0
    f_H1_C1 >= 0
    f_H1_C2 >= 0
    f_H1_C3 >= 0
    f_H1_C4 >= 0
    f_H2_C1 >= 0
    f_H2_C2 >= 0
    f_H2_C3 >= 0
    f_H2_C4 >= 0
    End
