Minimize
    obj: 9 x_1_A + 4 x_1_B + 3 x_1_C + 7 x_1_D + 6 x_1_E + 5 x_1_F + 6 x_1_G + 3 x_1_H + 7 x_1_I + 5 x_1_J + 4 x_2_A + 6 x_2_B + 5 x_2_C + 6 x_2_D + 4 x_2_E + 5 x_2_F + 3 x_2_G + 8 x_2_H + 7 x_2_I + 6 x_2_J + 5 x_3_A + 4 x_3_B + 7 x_3_C + 5 x_3_D + 6 x_3_E + 6 x_3_F + 5 x_3_G + 8 x_3_H + 6 x_3_I + 9 x_3_J + 7 x_4_A + 5 x_4_B + 2 x_4_C + 3 x_4_D + 7 x_4_E + 8 x_4_F + 5 x_4_G + 6 x_4_H + 8 x_4_I + 5 x_4_J + 10 x_5_A + 6 x_5_B + 7 x_5_C + 4 x_5_D + 5 x_5_E + 4 x_5_F + 4 x_5_G + 5 x_5_H + 9 x_5_I + 7 x_5_J + 6 x_6_A + 7 x_6_B + 6 x_6_C + 3 x_6_D + 9 x_6_E + 5 x_6_F + 7 x_6_G + 4 x_6_H + 3 x_6_I + 4 x_6_J + 8 x_7_A + 8 x_7_B + 5 x_7_C + 9 x_7_D + 5 x_7_E + 7 x_7_F + 5 x_7_G + 9 x_7_H + 5 x_7_I + 3 x_7_J + 7 x_8_A + 4 x_8_B + 8 x_8_C + 8 x_8_D + 6 x_8_E + 7 x_8_F + 5 x_8_G + 7 x_8_H + 7 x_8_I + 7 x_8_J + 5 x_9_A + 6 x_9_B + 8 x_9_C + 7 x_9_D + 7 x_9_E + 8 x_9_F + 7 x_9_G + 8 x_9_H + 4 x_9_I + 5 x_9_J + 8 x_10_A + 7 x_10_B + 9 x_10_C + 5 x_10_D + 8 x_10_E + 5 x_10_F + 9 x_10_G + 9 x_10_H + 3 x_10_I + 4 x_10_J + 9 x_11_A + 8 x_11_B + 10 x_11_C + 8 x_11_D + 5 x_11_E + 4 x_11_F + 7 x_11_G + 6 x_11_H + 8 x_11_I + 7 x_11_J + 8 x_12_A + 5 x_12_B + 6 x_12_C + 9 x_12_D + 4 x_12_E + 7 x_12_F + 8 x_12_G + 4 x_12_H + 7 x_12_I + 9 x_12_J
Subject To
    assign_task_A: x_1_A + x_2_A + x_3_A + x_4_A + x_5_A + x_6_A + x_7_A + x_8_A + x_9_A + x_10_A + x_11_A + x_12_A = 1
    assign_task_B: x_1_B + x_2_B + x_3_B + x_4_B + x_5_B + x_6_B + x_7_B + x_8_B + x_9_B + x_10_B + x_11_B + x_12_B = 1
    assign_task_C: x_1_C + x_2_C + x_3_C + x_4_C + x_5_C + x_6_C + x_7_C + x_8_C + x_9_C + x_10_C + x_11_C + x_12_C = 1
    assign_task_D: x_1_D + x_2_D + x_3_D + x_4_D + x_5_D + x_6_D + x_7_D + x_8_D + x_9_D + x_10_D + x_11_D + x_12_D = 1
    assign_task_E: x_1_E + x_2_E + x_3_E + x_4_E + x_5_E + x_6_E + x_7_E + x_8_E + x_9_E + x_10_E + x_11_E + x_12_E = 1
    assign_task_F: x_1_F + x_2_F + x_3_F + x_4_F + x_5_F + x_6_F + x_7_F + x_8_F + x_9_F + x_10_F + x_11_F + x_12_F = 1
    assign_task_G: x_1_G + x_2_G + x_3_G + x_4_G + x_5_G + x_6_G + x_7_G + x_8_G + x_9_G + x_10_G + x_11_G + x_12_G = 1
    assign_task_H: x_1_H + x_2_H + x_3_H + x_4_H + x_5_H + x_6_H + x_7_H + x_8_H + x_9_H + x_10_H + x_11_H + x_12_H = 1
    assign_task_I: x_1_I + x_2_I + x_3_I + x_4_I + x_5_I + x_6_I + x_7_I + x_8_I + x_9_I + x_10_I + x_11_I + x_12_I = 1
    assign_task_J: x_1_J + x_2_J + x_3_J + x_4_J + x_5_J + x_6_J + x_7_J + x_8_J + x_9_J + x_10_J + x_11_J + x_12_J = 1
    worker_1_limit: x_1_A + x_1_B + x_1_C + x_1_D + x_1_E + x_1_F + x_1_G + x_1_H + x_1_I + x_1_J <= 1
    worker_2_limit: x_2_A + x_2_B + x_2_C + x_2_D + x_2_E + x_2_F + x_2_G + x_2_H + x_2_I + x_2_J <= 1
    worker_3_limit: x_3_A + x_3_B + x_3_C + x_3_D + x_3_E + x_3_F + x_3_G + x_3_H + x_3_I + x_3_J <= 1
    worker_4_limit: x_4_A + x_4_B + x_4_C + x_4_D + x_4_E + x_4_F + x_4_G + x_4_H + x_4_I + x_4_J <= 1
    worker_5_limit: x_5_A + x_5_B + x_5_C + x_5_D + x_5_E + x_5_F + x_5_G + x_5_H + x_5_I + x_5_J <= 1
    worker_6_limit: x_6_A + x_6_B + x_6_C + x_6_D + x_6_E + x_6_F + x_6_G + x_6_H + x_6_I + x_6_J <= 1
    worker_7_limit: x_7_A + x_7_B + x_7_C + x_7_D + x_7_E + x_7_F + x_7_G + x_7_H + x_7_I + x_7_J <= 1
    worker_8_limit: x_8_A + x_8_B + x_8_C + x_8_D + x_8_E + x_8_F + x_8_G + x_8_H + x_8_I + x_8_J <= 1
    worker_9_limit: x_9_A + x_9_B + x_9_C + x_9_D + x_9_E + x_9_F + x_9_G + x_9_H + x_9_I + x_9_J <= 1
    worker_10_limit: x_10_A + x_10_B + x_10_C + x_10_D + x_10_E + x_10_F + x_10_G + x_10_H + x_10_I + x_10_J <= 1
    worker_11_limit: x_11_A + x_11_B + x_11_C + x_11_D + x_11_E + x_11_F + x_11_G + x_11_H + x_11_I + x_11_J <= 1
    worker_12_limit: x_12_A + x_12_B + x_12_C + x_12_D + x_12_E + x_12_F + x_12_G + x_12_H + x_12_I + x_12_J <= 1
    total_assignments: x_1_A + x_1_B + x_1_C + x_1_D + x_1_E + x_1_F + x_1_G + x_1_H + x_1_I + x_1_J + x_2_A + x_2_B + x_2_C + x_2_D + x_2_E + x_2_F + x_2_G + x_2_H + x_2_I + x_2_J + x_3_A + x_3_B + x_3_C + x_3_D + x_3_E + x_3_F + x_3_G + x_3_H + x_3_I + x_3_J + x_4_A + x_4_B + x_4_C + x_4_D + x_4_E + x_4_F + x_4_G + x_4_H + x_4_I + x_4_J + x_5_A + x_5_B + x_5_C + x_5_D + x_5_E + x_5_F + x_5_G + x_5_H + x_5_I + x_5_J + x_6_A + x_6_B + x_6_C + x_6_D + x_6_E + x_6_F + x_6_G + x_6_H + x_6_I + x_6_J + x_7_A + x_7_B + x_7_C + x_7_D + x_7_E + x_7_F + x_7_G + x_7_H + x_7_I + x_7_J + x_8_A + x_8_B + x_8_C + x_8_D + x_8_E + x_8_F + x_8_G + x_8_H + x_8_I + x_8_J + x_9_A + x_9_B + x_9_C + x_9_D + x_9_E + x_9_F + x_9_G + x_9_H + x_9_I + x_9_J + x_10_A + x_10_B + x_10_C + x_10_D + x_10_E + x_10_F + x_10_G + x_10_H + x_10_I + x_10_J + x_11_A + x_11_B + x_11_C + x_11_D + x_11_E + x_11_F + x_11_G + x_11_H + x_11_I + x_11_J + x_12_A + x_12_B + x_12_C + x_12_D + x_12_E + x_12_F + x_12_G + x_12_H + x_12_I + x_12_J = 10
Binaries
    x_1_A x_1_B x_1_C x_1_D x_1_E x_1_F x_1_G x_1_H x_1_I x_1_J x_2_A x_2_B x_2_C x_2_D x_2_E x_2_F x_2_G x_2_H x_2_I x_2_J x_3_A x_3_B x_3_C x_3_D x_3_E x_3_F x_3_G x_3_H x_3_I x_3_J x_4_A x_4_B x_4_C x_4_D x_4_E x_4_F x_4_G x_4_H x_4_I x_4_J x_5_A x_5_B x_5_C x_5_D x_5_E x_5_F x_5_G x_5_H x_5_I x_5_J x_6_A x_6_B x_6_C x_6_D x_6_E x_6_F x_6_G x_6_H x_6_I x_6_J x_7_A x_7_B x_7_C x_7_D x_7_E x_7_F x_7_G x_7_H x_7_I x_7_J x_8_A x_8_B x_8_C x_8_D x_8_E x_8_F x_8_G x_8_H x_8_I x_8_J x_9_A x_9_B x_9_C x_9_D x_9_E x_9_F x_9_G x_9_H x_9_I x_9_J x_10_A x_10_B x_10_C x_10_D x_10_E x_10_F x_10_G x_10_H x_10_I x_10_J x_11_A x_11_B x_11_C x_11_D x_11_E x_11_F x_11_G x_11_H x_11_I x_11_J x_12_A x_12_B x_12_C x_12_D x_12_E x_12_F x_12_G x_12_H x_12_I x_12_J
End
