\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 2 w_1_1 + 2 w_1_2 + 2 w_1_3 + 2 w_1_4 + 500 y_1_1 + 500 u_1_2 + 500 u_1_3 + 500 u_1_4 + 3 w_2_1 + 3 w_2_2 + 3 w_2_3 + 3 w_2_4 + 300 y_2_1 + 300 u_2_2 + 300 u_2_3 + 300 u_2_4 + 2.5 w_3_1 + 2.5 w_3_2 + 2.5 w_3_3 + 2.5 w_3_4 + 400 y_3_1 + 400 u_3_2 + 400 u_3_3 + 400 u_3_4 + 3 w_4_1 + 3 w_4_2 + 3 w_4_3 + 3 w_4_4 + 250 y_4_1 + 250 u_4_2 + 250 u_4_3 + 250 u_4_4 + 2.2000000000000002 w_5_1 + 2.2000000000000002 w_5_2 + 2.2000000000000002 w_5_3 + 2.2000000000000002 w_5_4 + 450 y_5_1 + 450 u_5_2 + 450 u_5_3 + 450 u_5_4 + 2.7999999999999998 w_6_1 + 2.7999999999999998 w_6_2 + 2.7999999999999998 w_6_3 + 2.7999999999999998 w_6_4 + 280 y_6_1 + 280 u_6_2 + 280 u_6_3 + 280 u_6_4 + 2.3999999999999999 w_7_1 + 2.3999999999999999 w_7_2 + 2.3999999999999999 w_7_3 + 2.3999999999999999 w_7_4 + 420 y_7_1 + 420 u_7_2 + 420 u_7_3 + 420 u_7_4
    + 3.2000000000000002 w_8_1 + 3.2000000000000002 w_8_2 + 3.2000000000000002 w_8_3 + 3.2000000000000002 w_8_4 + 200 y_8_1 + 200 u_8_2 + 200 u_8_3 + 200 u_8_4 + 2.1000000000000001 w_9_1 + 2.1000000000000001 w_9_2 + 2.1000000000000001 w_9_3 + 2.1000000000000001 w_9_4 + 480 y_9_1 + 480 u_9_2 + 480 u_9_3 + 480 u_9_4 + 2.8999999999999999 w_10_1 + 2.8999999999999999 w_10_2 + 2.8999999999999999 w_10_3 + 2.8999999999999999 w_10_4 + 260 y_10_1 + 260 u_10_2 + 260 u_10_3 + 260 u_10_4
Subject To
  demand_1: + 1 w_1_1 + 1 w_2_1 + 1 w_3_1 + 1 w_4_1 + 1 w_5_1 + 1 w_6_1 + 1 w_7_1 + 1 w_8_1 + 1 w_9_1 + 1 w_10_1 >= 1500
  demand_2: + 1 w_1_2 + 1 w_2_2 + 1 w_3_2 + 1 w_4_2 + 1 w_5_2 + 1 w_6_2 + 1 w_7_2 + 1 w_8_2 + 1 w_9_2 + 1 w_10_2 >= 2000
  demand_3: + 1 w_1_3 + 1 w_2_3 + 1 w_3_3 + 1 w_4_3 + 1 w_5_3 + 1 w_6_3 + 1 w_7_3 + 1 w_8_3 + 1 w_9_3 + 1 w_10_3 >= 1800
  demand_4: + 1 w_1_4 + 1 w_2_4 + 1 w_3_4 + 1 w_4_4 + 1 w_5_4 + 1 w_6_4 + 1 w_7_4 + 1 w_8_4 + 1 w_9_4 + 1 w_10_4 >= 1000
  capacity_link_1_1: + 1 w_1_1 - 1000 y_1_1 <= 0
  capacity_link_1_2: + 1 w_1_2 - 1000 y_1_2 <= 0
  capacity_link_1_3: + 1 w_1_3 - 1000 y_1_3 <= 0
  capacity_link_1_4: + 1 w_1_4 - 1000 y_1_4 <= 0
  capacity_link_2_1: + 1 w_2_1 - 800 y_2_1 <= 0
  capacity_link_2_2: + 1 w_2_2 - 800 y_2_2 <= 0
  capacity_link_2_3: + 1 w_2_3 - 800 y_2_3 <= 0
  capacity_link_2_4: + 1 w_2_4 - 800 y_2_4 <= 0
  capacity_link_3_1: + 1 w_3_1 - 1200 y_3_1 <= 0
  capacity_link_3_2: + 1 w_3_2 - 1200 y_3_2 <= 0
  capacity_link_3_3: + 1 w_3_3 - 1200 y_3_3 <= 0
  capacity_link_3_4: + 1 w_3_4 - 1200 y_3_4 <= 0
  capacity_link_4_1: + 1 w_4_1 - 600 y_4_1 <= 0
  capacity_link_4_2: + 1 w_4_2 - 600 y_4_2 <= 0
  capacity_link_4_3: + 1 w_4_3 - 600 y_4_3 <= 0
  capacity_link_4_4: + 1 w_4_4 - 600 y_4_4 <= 0
  capacity_link_5_1: + 1 w_5_1 - 900 y_5_1 <= 0
  capacity_link_5_2: + 1 w_5_2 - 900 y_5_2 <= 0
  capacity_link_5_3: + 1 w_5_3 - 900 y_5_3 <= 0
  capacity_link_5_4: + 1 w_5_4 - 900 y_5_4 <= 0
  capacity_link_6_1: + 1 w_6_1 - 700 y_6_1 <= 0
  capacity_link_6_2: + 1 w_6_2 - 700 y_6_2 <= 0
  capacity_link_6_3: + 1 w_6_3 - 700 y_6_3 <= 0
  capacity_link_6_4: + 1 w_6_4 - 700 y_6_4 <= 0
  capacity_link_7_1: + 1 w_7_1 - 1100 y_7_1 <= 0
  capacity_link_7_2: + 1 w_7_2 - 1100 y_7_2 <= 0
  capacity_link_7_3: + 1 w_7_3 - 1100 y_7_3 <= 0
  capacity_link_7_4: + 1 w_7_4 - 1100 y_7_4 <= 0
  capacity_link_8_1: + 1 w_8_1 - 500 y_8_1 <= 0
  capacity_link_8_2: + 1 w_8_2 - 500 y_8_2 <= 0
  capacity_link_8_3: + 1 w_8_3 - 500 y_8_3 <= 0
  capacity_link_8_4: + 1 w_8_4 - 500 y_8_4 <= 0
  capacity_link_9_1: + 1 w_9_1 - 1000 y_9_1 <= 0
  capacity_link_9_2: + 1 w_9_2 - 1000 y_9_2 <= 0
  capacity_link_9_3: + 1 w_9_3 - 1000 y_9_3 <= 0
  capacity_link_9_4: + 1 w_9_4 - 1000 y_9_4 <= 0
  capacity_link_10_1: + 1 w_10_1 - 650 y_10_1 <= 0
  capacity_link_10_2: + 1 w_10_2 - 650 y_10_2 <= 0
  capacity_link_10_3: + 1 w_10_3 - 650 y_10_3 <= 0
  capacity_link_10_4: + 1 w_10_4 - 650 y_10_4 <= 0
  spare_capacity_1: + 1 w_1_1 - 900 y_1_1 + 1 w_2_1 - 720 y_2_1 + 1 w_3_1 - 1080 y_3_1 + 1 w_4_1 - 540 y_4_1 + 1 w_5_1 - 810 y_5_1 + 1 w_6_1 - 630 y_6_1 + 1 w_7_1 - 990 y_7_1 + 1 w_8_1 - 450 y_8_1 + 1 w_9_1 - 900 y_9_1 + 1 w_10_1 - 585 y_10_1 <= 0
  spare_capacity_2: + 1 w_1_2 + 1 w_2_2 + 1 w_3_2 + 1 w_4_2 + 1 w_5_2 + 1 w_6_2 + 1 w_7_2 + 1 w_8_2 + 1 w_9_2 + 1 w_10_2 - 900 y_1_2 - 720 y_2_2 - 1080 y_3_2 - 540 y_4_2 - 810 y_5_2 - 630 y_6_2 - 990 y_7_2 - 450 y_8_2 - 900 y_9_2 - 585 y_10_2 <= 0
  spare_capacity_3: + 1 w_1_3 + 1 w_2_3 + 1 w_3_3 + 1 w_4_3 + 1 w_5_3 + 1 w_6_3 + 1 w_7_3 + 1 w_8_3 + 1 w_9_3 + 1 w_10_3 - 900 y_1_3 - 720 y_2_3 - 1080 y_3_3 - 540 y_4_3 - 810 y_5_3 - 630 y_6_3 - 990 y_7_3 - 450 y_8_3 - 900 y_9_3 - 585 y_10_3 <= 0
  spare_capacity_4: + 1 w_1_4 + 1 w_2_4 + 1 w_3_4 + 1 w_4_4 + 1 w_5_4 + 1 w_6_4 + 1 w_7_4 + 1 w_8_4 + 1 w_9_4 + 1 w_10_4 - 900 y_1_4 - 720 y_2_4 - 1080 y_3_4 - 540 y_4_4 - 810 y_5_4 - 630 y_6_4 - 990 y_7_4 - 450 y_8_4 - 900 y_9_4 - 585 y_10_4 <= 0
  startup_lb_1_2: - 1 y_1_1 - 1 u_1_2 + 1 y_1_2 <= 0
  startup_ub_status_1_2: + 1 u_1_2 - 1 y_1_2 <= 0
  startup_ub_prev_1_2: + 1 y_1_1 + 1 u_1_2 <= 1
  startup_lb_1_3: - 1 u_1_3 - 1 y_1_2 + 1 y_1_3 <= 0
  startup_ub_status_1_3: + 1 u_1_3 - 1 y_1_3 <= 0
  startup_ub_prev_1_3: + 1 u_1_3 + 1 y_1_2 <= 1
  startup_lb_1_4: - 1 u_1_4 - 1 y_1_3 + 1 y_1_4 <= 0
  startup_ub_status_1_4: + 1 u_1_4 - 1 y_1_4 <= 0
  startup_ub_prev_1_4: + 1 u_1_4 + 1 y_1_3 <= 1
  min_runtime_start_1_1: + 1 y_1_1 - 1 y_1_2 <= 0
  min_runtime_1_2: + 2 u_1_2 - 1 y_1_2 - 1 y_1_3 <= 0
  min_runtime_1_3: + 2 u_1_3 - 1 y_1_3 - 1 y_1_4 <= 0
  cooldown_1_2: + 1 y_1_1 - 1 y_1_2 + 1 y_1_3 <= 1
  cooldown_1_3: + 1 y_1_2 - 1 y_1_3 + 1 y_1_4 <= 1
  ramp_up_1_2: - 1 w_1_1 + 1 w_1_2 <= 300
  ramp_down_1_2: + 1 w_1_1 - 1 w_1_2 <= 300
  ramp_up_1_3: - 1 w_1_2 + 1 w_1_3 <= 300
  ramp_down_1_3: + 1 w_1_2 - 1 w_1_3 <= 300
  ramp_up_1_4: - 1 w_1_3 + 1 w_1_4 <= 300
  ramp_down_1_4: + 1 w_1_3 - 1 w_1_4 <= 300
  startup_lb_2_2: - 1 y_2_1 - 1 u_2_2 + 1 y_2_2 <= 0
  startup_ub_status_2_2: + 1 u_2_2 - 1 y_2_2 <= 0
  startup_ub_prev_2_2: + 1 y_2_1 + 1 u_2_2 <= 1
  startup_lb_2_3: - 1 u_2_3 - 1 y_2_2 + 1 y_2_3 <= 0
  startup_ub_status_2_3: + 1 u_2_3 - 1 y_2_3 <= 0
  startup_ub_prev_2_3: + 1 u_2_3 + 1 y_2_2 <= 1
  startup_lb_2_4: - 1 u_2_4 - 1 y_2_3 + 1 y_2_4 <= 0
  startup_ub_status_2_4: + 1 u_2_4 - 1 y_2_4 <= 0
  startup_ub_prev_2_4: + 1 u_2_4 + 1 y_2_3 <= 1
  min_runtime_start_1_2: + 1 y_2_1 - 1 y_2_2 <= 0
  min_runtime_2_2: + 2 u_2_2 - 1 y_2_2 - 1 y_2_3 <= 0
  min_runtime_2_3: + 2 u_2_3 - 1 y_2_3 - 1 y_2_4 <= 0
  cooldown_2_2: + 1 y_2_1 - 1 y_2_2 + 1 y_2_3 <= 1
  cooldown_2_3: + 1 y_2_2 - 1 y_2_3 + 1 y_2_4 <= 1
  ramp_up_2_2: - 1 w_2_1 + 1 w_2_2 <= 300
  ramp_down_2_2: + 1 w_2_1 - 1 w_2_2 <= 300
  ramp_up_2_3: - 1 w_2_2 + 1 w_2_3 <= 300
  ramp_down_2_3: + 1 w_2_2 - 1 w_2_3 <= 300
  ramp_up_2_4: - 1 w_2_3 + 1 w_2_4 <= 300
  ramp_down_2_4: + 1 w_2_3 - 1 w_2_4 <= 300
  startup_lb_3_2: - 1 y_3_1 - 1 u_3_2 + 1 y_3_2 <= 0
  startup_ub_status_3_2: + 1 u_3_2 - 1 y_3_2 <= 0
  startup_ub_prev_3_2: + 1 y_3_1 + 1 u_3_2 <= 1
  startup_lb_3_3: - 1 u_3_3 - 1 y_3_2 + 1 y_3_3 <= 0
  startup_ub_status_3_3: + 1 u_3_3 - 1 y_3_3 <= 0
  startup_ub_prev_3_3: + 1 u_3_3 + 1 y_3_2 <= 1
  startup_lb_3_4: - 1 u_3_4 - 1 y_3_3 + 1 y_3_4 <= 0
  startup_ub_status_3_4: + 1 u_3_4 - 1 y_3_4 <= 0
  startup_ub_prev_3_4: + 1 u_3_4 + 1 y_3_3 <= 1
  min_runtime_start_1_3: + 1 y_3_1 - 1 y_3_2 <= 0
  min_runtime_3_2: + 2 u_3_2 - 1 y_3_2 - 1 y_3_3 <= 0
  min_runtime_3_3: + 2 u_3_3 - 1 y_3_3 - 1 y_3_4 <= 0
  cooldown_3_2: + 1 y_3_1 - 1 y_3_2 + 1 y_3_3 <= 1
  cooldown_3_3: + 1 y_3_2 - 1 y_3_3 + 1 y_3_4 <= 1
  ramp_up_3_2: - 1 w_3_1 + 1 w_3_2 <= 300
  ramp_down_3_2: + 1 w_3_1 - 1 w_3_2 <= 300
  ramp_up_3_3: - 1 w_3_2 + 1 w_3_3 <= 300
  ramp_down_3_3: + 1 w_3_2 - 1 w_3_3 <= 300
  ramp_up_3_4: - 1 w_3_3 + 1 w_3_4 <= 300
  ramp_down_3_4: + 1 w_3_3 - 1 w_3_4 <= 300
  startup_lb_4_2: - 1 y_4_1 - 1 u_4_2 + 1 y_4_2 <= 0
  startup_ub_status_4_2: + 1 u_4_2 - 1 y_4_2 <= 0
  startup_ub_prev_4_2: + 1 y_4_1 + 1 u_4_2 <= 1
  startup_lb_4_3: - 1 u_4_3 - 1 y_4_2 + 1 y_4_3 <= 0
  startup_ub_status_4_3: + 1 u_4_3 - 1 y_4_3 <= 0
  startup_ub_prev_4_3: + 1 u_4_3 + 1 y_4_2 <= 1
  startup_lb_4_4: - 1 u_4_4 - 1 y_4_3 + 1 y_4_4 <= 0
  startup_ub_status_4_4: + 1 u_4_4 - 1 y_4_4 <= 0
  startup_ub_prev_4_4: + 1 u_4_4 + 1 y_4_3 <= 1
  min_runtime_start_1_4: + 1 y_4_1 - 1 y_4_2 <= 0
  min_runtime_4_2: + 2 u_4_2 - 1 y_4_2 - 1 y_4_3 <= 0
  min_runtime_4_3: + 2 u_4_3 - 1 y_4_3 - 1 y_4_4 <= 0
  cooldown_4_2: + 1 y_4_1 - 1 y_4_2 + 1 y_4_3 <= 1
  cooldown_4_3: + 1 y_4_2 - 1 y_4_3 + 1 y_4_4 <= 1
  ramp_up_4_2: - 1 w_4_1 + 1 w_4_2 <= 300
  ramp_down_4_2: + 1 w_4_1 - 1 w_4_2 <= 300
  ramp_up_4_3: - 1 w_4_2 + 1 w_4_3 <= 300
  ramp_down_4_3: + 1 w_4_2 - 1 w_4_3 <= 300
  ramp_up_4_4: - 1 w_4_3 + 1 w_4_4 <= 300
  ramp_down_4_4: + 1 w_4_3 - 1 w_4_4 <= 300
  startup_lb_5_2: - 1 y_5_1 - 1 u_5_2 + 1 y_5_2 <= 0
  startup_ub_status_5_2: + 1 u_5_2 - 1 y_5_2 <= 0
  startup_ub_prev_5_2: + 1 y_5_1 + 1 u_5_2 <= 1
  startup_lb_5_3: - 1 u_5_3 - 1 y_5_2 + 1 y_5_3 <= 0
  startup_ub_status_5_3: + 1 u_5_3 - 1 y_5_3 <= 0
  startup_ub_prev_5_3: + 1 u_5_3 + 1 y_5_2 <= 1
  startup_lb_5_4: - 1 u_5_4 - 1 y_5_3 + 1 y_5_4 <= 0
  startup_ub_status_5_4: + 1 u_5_4 - 1 y_5_4 <= 0
  startup_ub_prev_5_4: + 1 u_5_4 + 1 y_5_3 <= 1
  min_runtime_start_1_5: + 1 y_5_1 - 1 y_5_2 <= 0
  min_runtime_5_2: + 2 u_5_2 - 1 y_5_2 - 1 y_5_3 <= 0
  min_runtime_5_3: + 2 u_5_3 - 1 y_5_3 - 1 y_5_4 <= 0
  cooldown_5_2: + 1 y_5_1 - 1 y_5_2 + 1 y_5_3 <= 1
  cooldown_5_3: + 1 y_5_2 - 1 y_5_3 + 1 y_5_4 <= 1
  ramp_up_5_2: - 1 w_5_1 + 1 w_5_2 <= 300
  ramp_down_5_2: + 1 w_5_1 - 1 w_5_2 <= 300
  ramp_up_5_3: - 1 w_5_2 + 1 w_5_3 <= 300
  ramp_down_5_3: + 1 w_5_2 - 1 w_5_3 <= 300
  ramp_up_5_4: - 1 w_5_3 + 1 w_5_4 <= 300
  ramp_down_5_4: + 1 w_5_3 - 1 w_5_4 <= 300
  startup_lb_6_2: - 1 y_6_1 - 1 u_6_2 + 1 y_6_2 <= 0
  startup_ub_status_6_2: + 1 u_6_2 - 1 y_6_2 <= 0
  startup_ub_prev_6_2: + 1 y_6_1 + 1 u_6_2 <= 1
  startup_lb_6_3: - 1 u_6_3 - 1 y_6_2 + 1 y_6_3 <= 0
  startup_ub_status_6_3: + 1 u_6_3 - 1 y_6_3 <= 0
  startup_ub_prev_6_3: + 1 u_6_3 + 1 y_6_2 <= 1
  startup_lb_6_4: - 1 u_6_4 - 1 y_6_3 + 1 y_6_4 <= 0
  startup_ub_status_6_4: + 1 u_6_4 - 1 y_6_4 <= 0
  startup_ub_prev_6_4: + 1 u_6_4 + 1 y_6_3 <= 1
  min_runtime_start_1_6: + 1 y_6_1 - 1 y_6_2 <= 0
  min_runtime_6_2: + 2 u_6_2 - 1 y_6_2 - 1 y_6_3 <= 0
  min_runtime_6_3: + 2 u_6_3 - 1 y_6_3 - 1 y_6_4 <= 0
  cooldown_6_2: + 1 y_6_1 - 1 y_6_2 + 1 y_6_3 <= 1
  cooldown_6_3: + 1 y_6_2 - 1 y_6_3 + 1 y_6_4 <= 1
  ramp_up_6_2: - 1 w_6_1 + 1 w_6_2 <= 300
  ramp_down_6_2: + 1 w_6_1 - 1 w_6_2 <= 300
  ramp_up_6_3: - 1 w_6_2 + 1 w_6_3 <= 300
  ramp_down_6_3: + 1 w_6_2 - 1 w_6_3 <= 300
  ramp_up_6_4: - 1 w_6_3 + 1 w_6_4 <= 300
  ramp_down_6_4: + 1 w_6_3 - 1 w_6_4 <= 300
  startup_lb_7_2: - 1 y_7_1 - 1 u_7_2 + 1 y_7_2 <= 0
  startup_ub_status_7_2: + 1 u_7_2 - 1 y_7_2 <= 0
  startup_ub_prev_7_2: + 1 y_7_1 + 1 u_7_2 <= 1
  startup_lb_7_3: - 1 u_7_3 - 1 y_7_2 + 1 y_7_3 <= 0
  startup_ub_status_7_3: + 1 u_7_3 - 1 y_7_3 <= 0
  startup_ub_prev_7_3: + 1 u_7_3 + 1 y_7_2 <= 1
  startup_lb_7_4: - 1 u_7_4 - 1 y_7_3 + 1 y_7_4 <= 0
  startup_ub_status_7_4: + 1 u_7_4 - 1 y_7_4 <= 0
  startup_ub_prev_7_4: + 1 u_7_4 + 1 y_7_3 <= 1
  min_runtime_start_1_7: + 1 y_7_1 - 1 y_7_2 <= 0
  min_runtime_7_2: + 2 u_7_2 - 1 y_7_2 - 1 y_7_3 <= 0
  min_runtime_7_3: + 2 u_7_3 - 1 y_7_3 - 1 y_7_4 <= 0
  cooldown_7_2: + 1 y_7_1 - 1 y_7_2 + 1 y_7_3 <= 1
  cooldown_7_3: + 1 y_7_2 - 1 y_7_3 + 1 y_7_4 <= 1
  ramp_up_7_2: - 1 w_7_1 + 1 w_7_2 <= 300
  ramp_down_7_2: + 1 w_7_1 - 1 w_7_2 <= 300
  ramp_up_7_3: - 1 w_7_2 + 1 w_7_3 <= 300
  ramp_down_7_3: + 1 w_7_2 - 1 w_7_3 <= 300
  ramp_up_7_4: - 1 w_7_3 + 1 w_7_4 <= 300
  ramp_down_7_4: + 1 w_7_3 - 1 w_7_4 <= 300
  startup_lb_8_2: - 1 y_8_1 - 1 u_8_2 + 1 y_8_2 <= 0
  startup_ub_status_8_2: + 1 u_8_2 - 1 y_8_2 <= 0
  startup_ub_prev_8_2: + 1 y_8_1 + 1 u_8_2 <= 1
  startup_lb_8_3: - 1 u_8_3 - 1 y_8_2 + 1 y_8_3 <= 0
  startup_ub_status_8_3: + 1 u_8_3 - 1 y_8_3 <= 0
  startup_ub_prev_8_3: + 1 u_8_3 + 1 y_8_2 <= 1
  startup_lb_8_4: - 1 u_8_4 - 1 y_8_3 + 1 y_8_4 <= 0
  startup_ub_status_8_4: + 1 u_8_4 - 1 y_8_4 <= 0
  startup_ub_prev_8_4: + 1 u_8_4 + 1 y_8_3 <= 1
  min_runtime_start_1_8: + 1 y_8_1 - 1 y_8_2 <= 0
  min_runtime_8_2: + 2 u_8_2 - 1 y_8_2 - 1 y_8_3 <= 0
  min_runtime_8_3: + 2 u_8_3 - 1 y_8_3 - 1 y_8_4 <= 0
  cooldown_8_2: + 1 y_8_1 - 1 y_8_2 + 1 y_8_3 <= 1
  cooldown_8_3: + 1 y_8_2 - 1 y_8_3 + 1 y_8_4 <= 1
  ramp_up_8_2: - 1 w_8_1 + 1 w_8_2 <= 300
  ramp_down_8_2: + 1 w_8_1 - 1 w_8_2 <= 300
  ramp_up_8_3: - 1 w_8_2 + 1 w_8_3 <= 300
  ramp_down_8_3: + 1 w_8_2 - 1 w_8_3 <= 300
  ramp_up_8_4: - 1 w_8_3 + 1 w_8_4 <= 300
  ramp_down_8_4: + 1 w_8_3 - 1 w_8_4 <= 300
  startup_lb_9_2: - 1 y_9_1 - 1 u_9_2 + 1 y_9_2 <= 0
  startup_ub_status_9_2: + 1 u_9_2 - 1 y_9_2 <= 0
  startup_ub_prev_9_2: + 1 y_9_1 + 1 u_9_2 <= 1
  startup_lb_9_3: - 1 u_9_3 - 1 y_9_2 + 1 y_9_3 <= 0
  startup_ub_status_9_3: + 1 u_9_3 - 1 y_9_3 <= 0
  startup_ub_prev_9_3: + 1 u_9_3 + 1 y_9_2 <= 1
  startup_lb_9_4: - 1 u_9_4 - 1 y_9_3 + 1 y_9_4 <= 0
  startup_ub_status_9_4: + 1 u_9_4 - 1 y_9_4 <= 0
  startup_ub_prev_9_4: + 1 u_9_4 + 1 y_9_3 <= 1
  min_runtime_start_1_9: + 1 y_9_1 - 1 y_9_2 <= 0
  min_runtime_9_2: + 2 u_9_2 - 1 y_9_2 - 1 y_9_3 <= 0
  min_runtime_9_3: + 2 u_9_3 - 1 y_9_3 - 1 y_9_4 <= 0
  cooldown_9_2: + 1 y_9_1 - 1 y_9_2 + 1 y_9_3 <= 1
  cooldown_9_3: + 1 y_9_2 - 1 y_9_3 + 1 y_9_4 <= 1
  ramp_up_9_2: - 1 w_9_1 + 1 w_9_2 <= 300
  ramp_down_9_2: + 1 w_9_1 - 1 w_9_2 <= 300
  ramp_up_9_3: - 1 w_9_2 + 1 w_9_3 <= 300
  ramp_down_9_3: + 1 w_9_2 - 1 w_9_3 <= 300
  ramp_up_9_4: - 1 w_9_3 + 1 w_9_4 <= 300
  ramp_down_9_4: + 1 w_9_3 - 1 w_9_4 <= 300
  startup_lb_10_2: - 1 y_10_1 - 1 u_10_2 + 1 y_10_2 <= 0
  startup_ub_status_10_2: + 1 u_10_2 - 1 y_10_2 <= 0
  startup_ub_prev_10_2: + 1 y_10_1 + 1 u_10_2 <= 1
  startup_lb_10_3: - 1 u_10_3 - 1 y_10_2 + 1 y_10_3 <= 0
  startup_ub_status_10_3: + 1 u_10_3 - 1 y_10_3 <= 0
  startup_ub_prev_10_3: + 1 u_10_3 + 1 y_10_2 <= 1
  startup_lb_10_4: - 1 u_10_4 - 1 y_10_3 + 1 y_10_4 <= 0
  startup_ub_status_10_4: + 1 u_10_4 - 1 y_10_4 <= 0
  startup_ub_prev_10_4: + 1 u_10_4 + 1 y_10_3 <= 1
  min_runtime_start_1_10: + 1 y_10_1 - 1 y_10_2 <= 0
  min_runtime_10_2: + 2 u_10_2 - 1 y_10_2 - 1 y_10_3 <= 0
  min_runtime_10_3: + 2 u_10_3 - 1 y_10_3 - 1 y_10_4 <= 0
  cooldown_10_2: + 1 y_10_1 - 1 y_10_2 + 1 y_10_3 <= 1
  cooldown_10_3: + 1 y_10_2 - 1 y_10_3 + 1 y_10_4 <= 1
  ramp_up_10_2: - 1 w_10_1 + 1 w_10_2 <= 300
  ramp_down_10_2: + 1 w_10_1 - 1 w_10_2 <= 300
  ramp_up_10_3: - 1 w_10_2 + 1 w_10_3 <= 300
  ramp_down_10_3: + 1 w_10_2 - 1 w_10_3 <= 300
  ramp_up_10_4: - 1 w_10_3 + 1 w_10_4 <= 300
  ramp_down_10_4: + 1 w_10_3 - 1 w_10_4 <= 300
  __wl_bound_00001_lower: + 1 w_1_1 >= 0
  __wl_bound_00002_lower: + 1 w_1_2 >= 0
  __wl_bound_00003_lower: + 1 w_1_3 >= 0
  __wl_bound_00004_lower: + 1 w_1_4 >= 0
  __wl_bound_00005_lower: + 1 y_1_1 >= 0
  __wl_bound_00005_upper: + 1 y_1_1 <= 1
  __wl_bound_00006_lower: + 1 u_1_2 >= 0
  __wl_bound_00006_upper: + 1 u_1_2 <= 1
  __wl_bound_00007_lower: + 1 u_1_3 >= 0
  __wl_bound_00007_upper: + 1 u_1_3 <= 1
  __wl_bound_00008_lower: + 1 u_1_4 >= 0
  __wl_bound_00008_upper: + 1 u_1_4 <= 0
  __wl_bound_00009_lower: + 1 w_2_1 >= 0
  __wl_bound_00010_lower: + 1 w_2_2 >= 0
  __wl_bound_00011_lower: + 1 w_2_3 >= 0
  __wl_bound_00012_lower: + 1 w_2_4 >= 0
  __wl_bound_00013_lower: + 1 y_2_1 >= 0
  __wl_bound_00013_upper: + 1 y_2_1 <= 1
  __wl_bound_00014_lower: + 1 u_2_2 >= 0
  __wl_bound_00014_upper: + 1 u_2_2 <= 1
  __wl_bound_00015_lower: + 1 u_2_3 >= 0
  __wl_bound_00015_upper: + 1 u_2_3 <= 1
  __wl_bound_00016_lower: + 1 u_2_4 >= 0
  __wl_bound_00016_upper: + 1 u_2_4 <= 0
  __wl_bound_00017_lower: + 1 w_3_1 >= 0
  __wl_bound_00018_lower: + 1 w_3_2 >= 0
  __wl_bound_00019_lower: + 1 w_3_3 >= 0
  __wl_bound_00020_lower: + 1 w_3_4 >= 0
  __wl_bound_00021_lower: + 1 y_3_1 >= 0
  __wl_bound_00021_upper: + 1 y_3_1 <= 1
  __wl_bound_00022_lower: + 1 u_3_2 >= 0
  __wl_bound_00022_upper: + 1 u_3_2 <= 1
  __wl_bound_00023_lower: + 1 u_3_3 >= 0
  __wl_bound_00023_upper: + 1 u_3_3 <= 1
  __wl_bound_00024_lower: + 1 u_3_4 >= 0
  __wl_bound_00024_upper: + 1 u_3_4 <= 0
  __wl_bound_00025_lower: + 1 w_4_1 >= 0
  __wl_bound_00026_lower: + 1 w_4_2 >= 0
  __wl_bound_00027_lower: + 1 w_4_3 >= 0
  __wl_bound_00028_lower: + 1 w_4_4 >= 0
  __wl_bound_00029_lower: + 1 y_4_1 >= 0
  __wl_bound_00029_upper: + 1 y_4_1 <= 1
  __wl_bound_00030_lower: + 1 u_4_2 >= 0
  __wl_bound_00030_upper: + 1 u_4_2 <= 1
  __wl_bound_00031_lower: + 1 u_4_3 >= 0
  __wl_bound_00031_upper: + 1 u_4_3 <= 1
  __wl_bound_00032_lower: + 1 u_4_4 >= 0
  __wl_bound_00032_upper: + 1 u_4_4 <= 0
  __wl_bound_00033_lower: + 1 w_5_1 >= 0
  __wl_bound_00034_lower: + 1 w_5_2 >= 0
  __wl_bound_00035_lower: + 1 w_5_3 >= 0
  __wl_bound_00036_lower: + 1 w_5_4 >= 0
  __wl_bound_00037_lower: + 1 y_5_1 >= 0
  __wl_bound_00037_upper: + 1 y_5_1 <= 1
  __wl_bound_00038_lower: + 1 u_5_2 >= 0
  __wl_bound_00038_upper: + 1 u_5_2 <= 1
  __wl_bound_00039_lower: + 1 u_5_3 >= 0
  __wl_bound_00039_upper: + 1 u_5_3 <= 1
  __wl_bound_00040_lower: + 1 u_5_4 >= 0
  __wl_bound_00040_upper: + 1 u_5_4 <= 0
  __wl_bound_00041_lower: + 1 w_6_1 >= 0
  __wl_bound_00042_lower: + 1 w_6_2 >= 0
  __wl_bound_00043_lower: + 1 w_6_3 >= 0
  __wl_bound_00044_lower: + 1 w_6_4 >= 0
  __wl_bound_00045_lower: + 1 y_6_1 >= 0
  __wl_bound_00045_upper: + 1 y_6_1 <= 1
  __wl_bound_00046_lower: + 1 u_6_2 >= 0
  __wl_bound_00046_upper: + 1 u_6_2 <= 1
  __wl_bound_00047_lower: + 1 u_6_3 >= 0
  __wl_bound_00047_upper: + 1 u_6_3 <= 1
  __wl_bound_00048_lower: + 1 u_6_4 >= 0
  __wl_bound_00048_upper: + 1 u_6_4 <= 0
  __wl_bound_00049_lower: + 1 w_7_1 >= 0
  __wl_bound_00050_lower: + 1 w_7_2 >= 0
  __wl_bound_00051_lower: + 1 w_7_3 >= 0
  __wl_bound_00052_lower: + 1 w_7_4 >= 0
  __wl_bound_00053_lower: + 1 y_7_1 >= 0
  __wl_bound_00053_upper: + 1 y_7_1 <= 1
  __wl_bound_00054_lower: + 1 u_7_2 >= 0
  __wl_bound_00054_upper: + 1 u_7_2 <= 1
  __wl_bound_00055_lower: + 1 u_7_3 >= 0
  __wl_bound_00055_upper: + 1 u_7_3 <= 1
  __wl_bound_00056_lower: + 1 u_7_4 >= 0
  __wl_bound_00056_upper: + 1 u_7_4 <= 0
  __wl_bound_00057_lower: + 1 w_8_1 >= 0
  __wl_bound_00058_lower: + 1 w_8_2 >= 0
  __wl_bound_00059_lower: + 1 w_8_3 >= 0
  __wl_bound_00060_lower: + 1 w_8_4 >= 0
  __wl_bound_00061_lower: + 1 y_8_1 >= 0
  __wl_bound_00061_upper: + 1 y_8_1 <= 1
  __wl_bound_00062_lower: + 1 u_8_2 >= 0
  __wl_bound_00062_upper: + 1 u_8_2 <= 1
  __wl_bound_00063_lower: + 1 u_8_3 >= 0
  __wl_bound_00063_upper: + 1 u_8_3 <= 1
  __wl_bound_00064_lower: + 1 u_8_4 >= 0
  __wl_bound_00064_upper: + 1 u_8_4 <= 0
  __wl_bound_00065_lower: + 1 w_9_1 >= 0
  __wl_bound_00066_lower: + 1 w_9_2 >= 0
  __wl_bound_00067_lower: + 1 w_9_3 >= 0
  __wl_bound_00068_lower: + 1 w_9_4 >= 0
  __wl_bound_00069_lower: + 1 y_9_1 >= 0
  __wl_bound_00069_upper: + 1 y_9_1 <= 1
  __wl_bound_00070_lower: + 1 u_9_2 >= 0
  __wl_bound_00070_upper: + 1 u_9_2 <= 1
  __wl_bound_00071_lower: + 1 u_9_3 >= 0
  __wl_bound_00071_upper: + 1 u_9_3 <= 1
  __wl_bound_00072_lower: + 1 u_9_4 >= 0
  __wl_bound_00072_upper: + 1 u_9_4 <= 0
  __wl_bound_00073_lower: + 1 w_10_1 >= 0
  __wl_bound_00074_lower: + 1 w_10_2 >= 0
  __wl_bound_00075_lower: + 1 w_10_3 >= 0
  __wl_bound_00076_lower: + 1 w_10_4 >= 0
  __wl_bound_00077_lower: + 1 y_10_1 >= 0
  __wl_bound_00077_upper: + 1 y_10_1 <= 1
  __wl_bound_00078_lower: + 1 u_10_2 >= 0
  __wl_bound_00078_upper: + 1 u_10_2 <= 1
  __wl_bound_00079_lower: + 1 u_10_3 >= 0
  __wl_bound_00079_upper: + 1 u_10_3 <= 1
  __wl_bound_00080_lower: + 1 u_10_4 >= 0
  __wl_bound_00080_upper: + 1 u_10_4 <= 0
  __wl_bound_00081_lower: + 1 y_1_2 >= 0
  __wl_bound_00081_upper: + 1 y_1_2 <= 1
  __wl_bound_00082_lower: + 1 y_1_3 >= 0
  __wl_bound_00082_upper: + 1 y_1_3 <= 1
  __wl_bound_00083_lower: + 1 y_1_4 >= 0
  __wl_bound_00083_upper: + 1 y_1_4 <= 1
  __wl_bound_00084_lower: + 1 y_2_2 >= 0
  __wl_bound_00084_upper: + 1 y_2_2 <= 1
  __wl_bound_00085_lower: + 1 y_2_3 >= 0
  __wl_bound_00085_upper: + 1 y_2_3 <= 1
  __wl_bound_00086_lower: + 1 y_2_4 >= 0
  __wl_bound_00086_upper: + 1 y_2_4 <= 1
  __wl_bound_00087_lower: + 1 y_3_2 >= 0
  __wl_bound_00087_upper: + 1 y_3_2 <= 1
  __wl_bound_00088_lower: + 1 y_3_3 >= 0
  __wl_bound_00088_upper: + 1 y_3_3 <= 1
  __wl_bound_00089_lower: + 1 y_3_4 >= 0
  __wl_bound_00089_upper: + 1 y_3_4 <= 1
  __wl_bound_00090_lower: + 1 y_4_2 >= 0
  __wl_bound_00090_upper: + 1 y_4_2 <= 1
  __wl_bound_00091_lower: + 1 y_4_3 >= 0
  __wl_bound_00091_upper: + 1 y_4_3 <= 1
  __wl_bound_00092_lower: + 1 y_4_4 >= 0
  __wl_bound_00092_upper: + 1 y_4_4 <= 1
  __wl_bound_00093_lower: + 1 y_5_2 >= 0
  __wl_bound_00093_upper: + 1 y_5_2 <= 1
  __wl_bound_00094_lower: + 1 y_5_3 >= 0
  __wl_bound_00094_upper: + 1 y_5_3 <= 1
  __wl_bound_00095_lower: + 1 y_5_4 >= 0
  __wl_bound_00095_upper: + 1 y_5_4 <= 1
  __wl_bound_00096_lower: + 1 y_6_2 >= 0
  __wl_bound_00096_upper: + 1 y_6_2 <= 1
  __wl_bound_00097_lower: + 1 y_6_3 >= 0
  __wl_bound_00097_upper: + 1 y_6_3 <= 1
  __wl_bound_00098_lower: + 1 y_6_4 >= 0
  __wl_bound_00098_upper: + 1 y_6_4 <= 1
  __wl_bound_00099_lower: + 1 y_7_2 >= 0
  __wl_bound_00099_upper: + 1 y_7_2 <= 1
  __wl_bound_00100_lower: + 1 y_7_3 >= 0
  __wl_bound_00100_upper: + 1 y_7_3 <= 1
  __wl_bound_00101_lower: + 1 y_7_4 >= 0
  __wl_bound_00101_upper: + 1 y_7_4 <= 1
  __wl_bound_00102_lower: + 1 y_8_2 >= 0
  __wl_bound_00102_upper: + 1 y_8_2 <= 1
  __wl_bound_00103_lower: + 1 y_8_3 >= 0
  __wl_bound_00103_upper: + 1 y_8_3 <= 1
  __wl_bound_00104_lower: + 1 y_8_4 >= 0
  __wl_bound_00104_upper: + 1 y_8_4 <= 1
  __wl_bound_00105_lower: + 1 y_9_2 >= 0
  __wl_bound_00105_upper: + 1 y_9_2 <= 1
  __wl_bound_00106_lower: + 1 y_9_3 >= 0
  __wl_bound_00106_upper: + 1 y_9_3 <= 1
  __wl_bound_00107_lower: + 1 y_9_4 >= 0
  __wl_bound_00107_upper: + 1 y_9_4 <= 1
  __wl_bound_00108_lower: + 1 y_10_2 >= 0
  __wl_bound_00108_upper: + 1 y_10_2 <= 1
  __wl_bound_00109_lower: + 1 y_10_3 >= 0
  __wl_bound_00109_upper: + 1 y_10_3 <= 1
  __wl_bound_00110_lower: + 1 y_10_4 >= 0
  __wl_bound_00110_upper: + 1 y_10_4 <= 1
Bounds
  w_1_1 free
  w_1_2 free
  w_1_3 free
  w_1_4 free
  w_2_1 free
  w_2_2 free
  w_2_3 free
  w_2_4 free
  w_3_1 free
  w_3_2 free
  w_3_3 free
  w_3_4 free
  w_4_1 free
  w_4_2 free
  w_4_3 free
  w_4_4 free
  w_5_1 free
  w_5_2 free
  w_5_3 free
  w_5_4 free
  w_6_1 free
  w_6_2 free
  w_6_3 free
  w_6_4 free
  w_7_1 free
  w_7_2 free
  w_7_3 free
  w_7_4 free
  w_8_1 free
  w_8_2 free
  w_8_3 free
  w_8_4 free
  w_9_1 free
  w_9_2 free
  w_9_3 free
  w_9_4 free
  w_10_1 free
  w_10_2 free
  w_10_3 free
  w_10_4 free
Binaries
  y_1_1 u_1_2 u_1_3 u_1_4 y_2_1 u_2_2 u_2_3 u_2_4 y_3_1 u_3_2 u_3_3 u_3_4 y_4_1 u_4_2 u_4_3 u_4_4 y_5_1 u_5_2 u_5_3 u_5_4 y_6_1 u_6_2 u_6_3 u_6_4 y_7_1 u_7_2 u_7_3 u_7_4 y_8_1 u_8_2 u_8_3 u_8_4 y_9_1 u_9_2 u_9_3 u_9_4 y_10_1 u_10_2 u_10_3 u_10_4 y_1_2 y_1_3 y_1_4 y_2_2 y_2_3 y_2_4 y_3_2 y_3_3 y_3_4 y_4_2 y_4_3 y_4_4 y_5_2 y_5_3 y_5_4 y_6_2 y_6_3 y_6_4 y_7_2 y_7_3 y_7_4 y_8_2 y_8_3 y_8_4 y_9_2 y_9_3 y_9_4 y_10_2 y_10_3 y_10_4
End
