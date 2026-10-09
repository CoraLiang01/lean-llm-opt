\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 2000 y[W1] + 2500 y[W2] + 1800 y[W3] + 3200 y[W4] + 1500 y[W5] + 4000 y[W6] + 2800 y[W7] + 1950 y[W8] + 3500 y[W9] + 2200 y[W10] + 10 s[W1,C1] + 15 s[W1,C2] + 20 s[W1,C3] + 11 s[W1,C4] + 16 s[W1,C5] + 18 s[W1,C6] + 7 s[W1,C7] + 12 s[W1,C8] + 22 s[W1,C9] + 9 s[W1,C10] + 14 s[W1,C11] + 19 s[W1,C12] + 25 s[W1,C13] + 13 s[W1,C14] + 17 s[W1,C15] + 6 s[W1,C16] + 21 s[W1,C17] + 15 s[W1,C18] + 8 s[W1,C19] + 10 s[W1,C20] + 18 s[W2,C1] + 12 s[W2,C2] + 9 s[W2,C3] + 14 s[W2,C4] + 10 s[W2,C5] + 5 s[W2,C6] + 19 s[W2,C7] + 23 s[W2,C8] + 11 s[W2,C9] + 16 s[W2,C10] + 20 s[W2,C11] + 8 s[W2,C12] + 15 s[W2,C13] + 22 s[W2,C14] + 7 s[W2,C15] + 13 s[W2,C16] + 24 s[W2,C17] + 17 s[W2,C18] + 12 s[W2,C19] + 6 s[W2,C20] + 13 s[W3,C1] + 17 s[W3,C2] + 15 s[W3,C3] + 8 s[W3,C4] + 12 s[W3,C5] + 21 s[W3,C6] + 16 s[W3,C7] + 10 s[W3,C8] + 5 s[W3,C9] + 24 s[W3,C10]
    + 13 s[W3,C11] + 22 s[W3,C12] + 7 s[W3,C13] + 19 s[W3,C14] + 14 s[W3,C15] + 18 s[W3,C16] + 9 s[W3,C17] + 25 s[W3,C18] + 11 s[W3,C19] + 16 s[W3,C20] + 7 s[W4,C1] + 22 s[W4,C2] + 11 s[W4,C3] + 16 s[W4,C4] + 20 s[W4,C5] + 8 s[W4,C6] + 15 s[W4,C7] + 19 s[W4,C8] + 13 s[W4,C9] + 25 s[W4,C10] + 6 s[W4,C11] + 14 s[W4,C12] + 21 s[W4,C13] + 9 s[W4,C14] + 23 s[W4,C15] + 17 s[W4,C16] + 10 s[W4,C17] + 18 s[W4,C18] + 24 s[W4,C19] + 5 s[W4,C20] + 16 s[W5,C1] + 9 s[W5,C2] + 25 s[W5,C3] + 13 s[W5,C4] + 7 s[W5,C5] + 10 s[W5,C6] + 23 s[W5,C7] + 14 s[W5,C8] + 18 s[W5,C9] + 21 s[W5,C10] + 5 s[W5,C11] + 17 s[W5,C12] + 9 s[W5,C13] + 24 s[W5,C14] + 12 s[W5,C15] + 20 s[W5,C16] + 6 s[W5,C17] + 15 s[W5,C18] + 19 s[W5,C19] + 11 s[W5,C20] + 22 s[W6,C1] + 6 s[W6,C2] + 14 s[W6,C3] + 19 s[W6,C4] + 23 s[W6,C5] + 11 s[W6,C6] + 8 s[W6,C7] + 17 s[W6,C8] + 9 s[W6,C9]
    + 12 s[W6,C10] + 15 s[W6,C11] + 24 s[W6,C12] + 5 s[W6,C13] + 20 s[W6,C14] + 10 s[W6,C15] + 25 s[W6,C16] + 13 s[W6,C17] + 7 s[W6,C18] + 18 s[W6,C19] + 16 s[W6,C20] + 8 s[W7,C1] + 25 s[W7,C2] + 17 s[W7,C3] + 9 s[W7,C4] + 14 s[W7,C5] + 22 s[W7,C6] + 11 s[W7,C7] + 6 s[W7,C8] + 16 s[W7,C9] + 20 s[W7,C10] + 18 s[W7,C11] + 13 s[W7,C12] + 24 s[W7,C13] + 5 s[W7,C14] + 19 s[W7,C15] + 12 s[W7,C16] + 23 s[W7,C17] + 10 s[W7,C18] + 7 s[W7,C19] + 15 s[W7,C20] + 19 s[W8,C1] + 11 s[W8,C2] + 7 s[W8,C3] + 21 s[W8,C4] + 15 s[W8,C5] + 24 s[W8,C6] + 13 s[W8,C7] + 16 s[W8,C8] + 20 s[W8,C9] + 8 s[W8,C10] + 17 s[W8,C11] + 10 s[W8,C12] + 12 s[W8,C13] + 23 s[W8,C14] + 5 s[W8,C15] + 14 s[W8,C16] + 22 s[W8,C17] + 9 s[W8,C18] + 16 s[W8,C19] + 25 s[W8,C20] + 12 s[W9,C1] + 20 s[W9,C2] + 5 s[W9,C3] + 23 s[W9,C4] + 17 s[W9,C5] + 14 s[W9,C6] + 9 s[W9,C7] + 25 s[W9,C8]
    + 18 s[W9,C9] + 11 s[W9,C10] + 16 s[W9,C11] + 21 s[W9,C12] + 10 s[W9,C13] + 7 s[W9,C14] + 24 s[W9,C15] + 15 s[W9,C16] + 19 s[W9,C17] + 6 s[W9,C18] + 13 s[W9,C19] + 22 s[W9,C20] + 25 s[W10,C1] + 14 s[W10,C2] + 22 s[W10,C3] + 5 s[W10,C4] + 19 s[W10,C5] + 12 s[W10,C6] + 24 s[W10,C7] + 7 s[W10,C8] + 15 s[W10,C9] + 17 s[W10,C10] + 23 s[W10,C11] + 6 s[W10,C12] + 16 s[W10,C13] + 10 s[W10,C14] + 20 s[W10,C15] + 9 s[W10,C16] + 18 s[W10,C17] + 11 s[W10,C18] + 25 s[W10,C19] + 14 s[W10,C20]
Subject To
  Demand_C1: + 1 s[W1,C1] + 1 s[W2,C1] + 1 s[W3,C1] + 1 s[W4,C1] + 1 s[W5,C1] + 1 s[W6,C1] + 1 s[W7,C1] + 1 s[W8,C1] + 1 s[W9,C1] + 1 s[W10,C1] = 800
  Demand_C2: + 1 s[W1,C2] + 1 s[W2,C2] + 1 s[W3,C2] + 1 s[W4,C2] + 1 s[W5,C2] + 1 s[W6,C2] + 1 s[W7,C2] + 1 s[W8,C2] + 1 s[W9,C2] + 1 s[W10,C2] = 600
  Demand_C3: + 1 s[W1,C3] + 1 s[W2,C3] + 1 s[W3,C3] + 1 s[W4,C3] + 1 s[W5,C3] + 1 s[W6,C3] + 1 s[W7,C3] + 1 s[W8,C3] + 1 s[W9,C3] + 1 s[W10,C3] = 500
  Demand_C4: + 1 s[W1,C4] + 1 s[W2,C4] + 1 s[W3,C4] + 1 s[W4,C4] + 1 s[W5,C4] + 1 s[W6,C4] + 1 s[W7,C4] + 1 s[W8,C4] + 1 s[W9,C4] + 1 s[W10,C4] = 700
  Demand_C5: + 1 s[W1,C5] + 1 s[W2,C5] + 1 s[W3,C5] + 1 s[W4,C5] + 1 s[W5,C5] + 1 s[W6,C5] + 1 s[W7,C5] + 1 s[W8,C5] + 1 s[W9,C5] + 1 s[W10,C5] = 450
  Demand_C6: + 1 s[W1,C6] + 1 s[W2,C6] + 1 s[W3,C6] + 1 s[W4,C6] + 1 s[W5,C6] + 1 s[W6,C6] + 1 s[W7,C6] + 1 s[W8,C6] + 1 s[W9,C6] + 1 s[W10,C6] = 950
  Demand_C7: + 1 s[W1,C7] + 1 s[W2,C7] + 1 s[W3,C7] + 1 s[W4,C7] + 1 s[W5,C7] + 1 s[W6,C7] + 1 s[W7,C7] + 1 s[W8,C7] + 1 s[W9,C7] + 1 s[W10,C7] = 350
  Demand_C8: + 1 s[W1,C8] + 1 s[W2,C8] + 1 s[W3,C8] + 1 s[W4,C8] + 1 s[W5,C8] + 1 s[W6,C8] + 1 s[W7,C8] + 1 s[W8,C8] + 1 s[W9,C8] + 1 s[W10,C8] = 850
  Demand_C9: + 1 s[W1,C9] + 1 s[W2,C9] + 1 s[W3,C9] + 1 s[W4,C9] + 1 s[W5,C9] + 1 s[W6,C9] + 1 s[W7,C9] + 1 s[W8,C9] + 1 s[W9,C9] + 1 s[W10,C9] = 400
  Demand_C10: + 1 s[W1,C10] + 1 s[W2,C10] + 1 s[W3,C10] + 1 s[W4,C10] + 1 s[W5,C10] + 1 s[W6,C10] + 1 s[W7,C10] + 1 s[W8,C10] + 1 s[W9,C10] + 1 s[W10,C10] = 750
  Demand_C11: + 1 s[W1,C11] + 1 s[W2,C11] + 1 s[W3,C11] + 1 s[W4,C11] + 1 s[W5,C11] + 1 s[W6,C11] + 1 s[W7,C11] + 1 s[W8,C11] + 1 s[W9,C11] + 1 s[W10,C11] = 900
  Demand_C12: + 1 s[W1,C12] + 1 s[W2,C12] + 1 s[W3,C12] + 1 s[W4,C12] + 1 s[W5,C12] + 1 s[W6,C12] + 1 s[W7,C12] + 1 s[W8,C12] + 1 s[W9,C12] + 1 s[W10,C12] = 550
  Demand_C13: + 1 s[W1,C13] + 1 s[W2,C13] + 1 s[W3,C13] + 1 s[W4,C13] + 1 s[W5,C13] + 1 s[W6,C13] + 1 s[W7,C13] + 1 s[W8,C13] + 1 s[W9,C13] + 1 s[W10,C13] = 650
  Demand_C14: + 1 s[W1,C14] + 1 s[W2,C14] + 1 s[W3,C14] + 1 s[W4,C14] + 1 s[W5,C14] + 1 s[W6,C14] + 1 s[W7,C14] + 1 s[W8,C14] + 1 s[W9,C14] + 1 s[W10,C14] = 820
  Demand_C15: + 1 s[W1,C15] + 1 s[W2,C15] + 1 s[W3,C15] + 1 s[W4,C15] + 1 s[W5,C15] + 1 s[W6,C15] + 1 s[W7,C15] + 1 s[W8,C15] + 1 s[W9,C15] + 1 s[W10,C15] = 480
  Demand_C16: + 1 s[W1,C16] + 1 s[W2,C16] + 1 s[W3,C16] + 1 s[W4,C16] + 1 s[W5,C16] + 1 s[W6,C16] + 1 s[W7,C16] + 1 s[W8,C16] + 1 s[W9,C16] + 1 s[W10,C16] = 920
  Demand_C17: + 1 s[W1,C17] + 1 s[W2,C17] + 1 s[W3,C17] + 1 s[W4,C17] + 1 s[W5,C17] + 1 s[W6,C17] + 1 s[W7,C17] + 1 s[W8,C17] + 1 s[W9,C17] + 1 s[W10,C17] = 320
  Demand_C18: + 1 s[W1,C18] + 1 s[W2,C18] + 1 s[W3,C18] + 1 s[W4,C18] + 1 s[W5,C18] + 1 s[W6,C18] + 1 s[W7,C18] + 1 s[W8,C18] + 1 s[W9,C18] + 1 s[W10,C18] = 780
  Demand_C19: + 1 s[W1,C19] + 1 s[W2,C19] + 1 s[W3,C19] + 1 s[W4,C19] + 1 s[W5,C19] + 1 s[W6,C19] + 1 s[W7,C19] + 1 s[W8,C19] + 1 s[W9,C19] + 1 s[W10,C19] = 520
  Demand_C20: + 1 s[W1,C20] + 1 s[W2,C20] + 1 s[W3,C20] + 1 s[W4,C20] + 1 s[W5,C20] + 1 s[W6,C20] + 1 s[W7,C20] + 1 s[W8,C20] + 1 s[W9,C20] + 1 s[W10,C20] = 680
  Capacity_W1: - 1000 y[W1] + 1 s[W1,C1] + 1 s[W1,C2] + 1 s[W1,C3] + 1 s[W1,C4] + 1 s[W1,C5] + 1 s[W1,C6] + 1 s[W1,C7] + 1 s[W1,C8] + 1 s[W1,C9] + 1 s[W1,C10] + 1 s[W1,C11] + 1 s[W1,C12] + 1 s[W1,C13] + 1 s[W1,C14] + 1 s[W1,C15] + 1 s[W1,C16] + 1 s[W1,C17] + 1 s[W1,C18] + 1 s[W1,C19] + 1 s[W1,C20] <= 0
  Capacity_W2: - 1500 y[W2] + 1 s[W2,C1] + 1 s[W2,C2] + 1 s[W2,C3] + 1 s[W2,C4] + 1 s[W2,C5] + 1 s[W2,C6] + 1 s[W2,C7] + 1 s[W2,C8] + 1 s[W2,C9] + 1 s[W2,C10] + 1 s[W2,C11] + 1 s[W2,C12] + 1 s[W2,C13] + 1 s[W2,C14] + 1 s[W2,C15] + 1 s[W2,C16] + 1 s[W2,C17] + 1 s[W2,C18] + 1 s[W2,C19] + 1 s[W2,C20] <= 0
  Capacity_W3: - 1200 y[W3] + 1 s[W3,C1] + 1 s[W3,C2] + 1 s[W3,C3] + 1 s[W3,C4] + 1 s[W3,C5] + 1 s[W3,C6] + 1 s[W3,C7] + 1 s[W3,C8] + 1 s[W3,C9] + 1 s[W3,C10] + 1 s[W3,C11] + 1 s[W3,C12] + 1 s[W3,C13] + 1 s[W3,C14] + 1 s[W3,C15] + 1 s[W3,C16] + 1 s[W3,C17] + 1 s[W3,C18] + 1 s[W3,C19] + 1 s[W3,C20] <= 0
  Capacity_W4: - 2000 y[W4] + 1 s[W4,C1] + 1 s[W4,C2] + 1 s[W4,C3] + 1 s[W4,C4] + 1 s[W4,C5] + 1 s[W4,C6] + 1 s[W4,C7] + 1 s[W4,C8] + 1 s[W4,C9] + 1 s[W4,C10] + 1 s[W4,C11] + 1 s[W4,C12] + 1 s[W4,C13] + 1 s[W4,C14] + 1 s[W4,C15] + 1 s[W4,C16] + 1 s[W4,C17] + 1 s[W4,C18] + 1 s[W4,C19] + 1 s[W4,C20] <= 0
  Capacity_W5: - 800 y[W5] + 1 s[W5,C1] + 1 s[W5,C2] + 1 s[W5,C3] + 1 s[W5,C4] + 1 s[W5,C5] + 1 s[W5,C6] + 1 s[W5,C7] + 1 s[W5,C8] + 1 s[W5,C9] + 1 s[W5,C10] + 1 s[W5,C11] + 1 s[W5,C12] + 1 s[W5,C13] + 1 s[W5,C14] + 1 s[W5,C15] + 1 s[W5,C16] + 1 s[W5,C17] + 1 s[W5,C18] + 1 s[W5,C19] + 1 s[W5,C20] <= 0
  Capacity_W6: - 2500 y[W6] + 1 s[W6,C1] + 1 s[W6,C2] + 1 s[W6,C3] + 1 s[W6,C4] + 1 s[W6,C5] + 1 s[W6,C6] + 1 s[W6,C7] + 1 s[W6,C8] + 1 s[W6,C9] + 1 s[W6,C10] + 1 s[W6,C11] + 1 s[W6,C12] + 1 s[W6,C13] + 1 s[W6,C14] + 1 s[W6,C15] + 1 s[W6,C16] + 1 s[W6,C17] + 1 s[W6,C18] + 1 s[W6,C19] + 1 s[W6,C20] <= 0
  Capacity_W7: - 1800 y[W7] + 1 s[W7,C1] + 1 s[W7,C2] + 1 s[W7,C3] + 1 s[W7,C4] + 1 s[W7,C5] + 1 s[W7,C6] + 1 s[W7,C7] + 1 s[W7,C8] + 1 s[W7,C9] + 1 s[W7,C10] + 1 s[W7,C11] + 1 s[W7,C12] + 1 s[W7,C13] + 1 s[W7,C14] + 1 s[W7,C15] + 1 s[W7,C16] + 1 s[W7,C17] + 1 s[W7,C18] + 1 s[W7,C19] + 1 s[W7,C20] <= 0
  Capacity_W8: - 1100 y[W8] + 1 s[W8,C1] + 1 s[W8,C2] + 1 s[W8,C3] + 1 s[W8,C4] + 1 s[W8,C5] + 1 s[W8,C6] + 1 s[W8,C7] + 1 s[W8,C8] + 1 s[W8,C9] + 1 s[W8,C10] + 1 s[W8,C11] + 1 s[W8,C12] + 1 s[W8,C13] + 1 s[W8,C14] + 1 s[W8,C15] + 1 s[W8,C16] + 1 s[W8,C17] + 1 s[W8,C18] + 1 s[W8,C19] + 1 s[W8,C20] <= 0
  Capacity_W9: - 2100 y[W9] + 1 s[W9,C1] + 1 s[W9,C2] + 1 s[W9,C3] + 1 s[W9,C4] + 1 s[W9,C5] + 1 s[W9,C6] + 1 s[W9,C7] + 1 s[W9,C8] + 1 s[W9,C9] + 1 s[W9,C10] + 1 s[W9,C11] + 1 s[W9,C12] + 1 s[W9,C13] + 1 s[W9,C14] + 1 s[W9,C15] + 1 s[W9,C16] + 1 s[W9,C17] + 1 s[W9,C18] + 1 s[W9,C19] + 1 s[W9,C20] <= 0
  Capacity_W10: - 1300 y[W10] + 1 s[W10,C1] + 1 s[W10,C2] + 1 s[W10,C3] + 1 s[W10,C4] + 1 s[W10,C5] + 1 s[W10,C6] + 1 s[W10,C7] + 1 s[W10,C8] + 1 s[W10,C9] + 1 s[W10,C10] + 1 s[W10,C11] + 1 s[W10,C12] + 1 s[W10,C13] + 1 s[W10,C14] + 1 s[W10,C15] + 1 s[W10,C16] + 1 s[W10,C17] + 1 s[W10,C18] + 1 s[W10,C19] + 1 s[W10,C20] <= 0
  __wl_bound_00001_lower: + 1 y[W1] >= 0
  __wl_bound_00001_upper: + 1 y[W1] <= 1
  __wl_bound_00002_lower: + 1 y[W2] >= 0
  __wl_bound_00002_upper: + 1 y[W2] <= 1
  __wl_bound_00003_lower: + 1 y[W3] >= 0
  __wl_bound_00003_upper: + 1 y[W3] <= 1
  __wl_bound_00004_lower: + 1 y[W4] >= 0
  __wl_bound_00004_upper: + 1 y[W4] <= 1
  __wl_bound_00005_lower: + 1 y[W5] >= 0
  __wl_bound_00005_upper: + 1 y[W5] <= 1
  __wl_bound_00006_lower: + 1 y[W6] >= 0
  __wl_bound_00006_upper: + 1 y[W6] <= 1
  __wl_bound_00007_lower: + 1 y[W7] >= 0
  __wl_bound_00007_upper: + 1 y[W7] <= 1
  __wl_bound_00008_lower: + 1 y[W8] >= 0
  __wl_bound_00008_upper: + 1 y[W8] <= 1
  __wl_bound_00009_lower: + 1 y[W9] >= 0
  __wl_bound_00009_upper: + 1 y[W9] <= 1
  __wl_bound_00010_lower: + 1 y[W10] >= 0
  __wl_bound_00010_upper: + 1 y[W10] <= 1
  __wl_bound_00011_lower: + 1 s[W1,C1] >= 0
  __wl_bound_00012_lower: + 1 s[W1,C2] >= 0
  __wl_bound_00013_lower: + 1 s[W1,C3] >= 0
  __wl_bound_00014_lower: + 1 s[W1,C4] >= 0
  __wl_bound_00015_lower: + 1 s[W1,C5] >= 0
  __wl_bound_00016_lower: + 1 s[W1,C6] >= 0
  __wl_bound_00017_lower: + 1 s[W1,C7] >= 0
  __wl_bound_00018_lower: + 1 s[W1,C8] >= 0
  __wl_bound_00019_lower: + 1 s[W1,C9] >= 0
  __wl_bound_00020_lower: + 1 s[W1,C10] >= 0
  __wl_bound_00021_lower: + 1 s[W1,C11] >= 0
  __wl_bound_00022_lower: + 1 s[W1,C12] >= 0
  __wl_bound_00023_lower: + 1 s[W1,C13] >= 0
  __wl_bound_00024_lower: + 1 s[W1,C14] >= 0
  __wl_bound_00025_lower: + 1 s[W1,C15] >= 0
  __wl_bound_00026_lower: + 1 s[W1,C16] >= 0
  __wl_bound_00027_lower: + 1 s[W1,C17] >= 0
  __wl_bound_00028_lower: + 1 s[W1,C18] >= 0
  __wl_bound_00029_lower: + 1 s[W1,C19] >= 0
  __wl_bound_00030_lower: + 1 s[W1,C20] >= 0
  __wl_bound_00031_lower: + 1 s[W2,C1] >= 0
  __wl_bound_00032_lower: + 1 s[W2,C2] >= 0
  __wl_bound_00033_lower: + 1 s[W2,C3] >= 0
  __wl_bound_00034_lower: + 1 s[W2,C4] >= 0
  __wl_bound_00035_lower: + 1 s[W2,C5] >= 0
  __wl_bound_00036_lower: + 1 s[W2,C6] >= 0
  __wl_bound_00037_lower: + 1 s[W2,C7] >= 0
  __wl_bound_00038_lower: + 1 s[W2,C8] >= 0
  __wl_bound_00039_lower: + 1 s[W2,C9] >= 0
  __wl_bound_00040_lower: + 1 s[W2,C10] >= 0
  __wl_bound_00041_lower: + 1 s[W2,C11] >= 0
  __wl_bound_00042_lower: + 1 s[W2,C12] >= 0
  __wl_bound_00043_lower: + 1 s[W2,C13] >= 0
  __wl_bound_00044_lower: + 1 s[W2,C14] >= 0
  __wl_bound_00045_lower: + 1 s[W2,C15] >= 0
  __wl_bound_00046_lower: + 1 s[W2,C16] >= 0
  __wl_bound_00047_lower: + 1 s[W2,C17] >= 0
  __wl_bound_00048_lower: + 1 s[W2,C18] >= 0
  __wl_bound_00049_lower: + 1 s[W2,C19] >= 0
  __wl_bound_00050_lower: + 1 s[W2,C20] >= 0
  __wl_bound_00051_lower: + 1 s[W3,C1] >= 0
  __wl_bound_00052_lower: + 1 s[W3,C2] >= 0
  __wl_bound_00053_lower: + 1 s[W3,C3] >= 0
  __wl_bound_00054_lower: + 1 s[W3,C4] >= 0
  __wl_bound_00055_lower: + 1 s[W3,C5] >= 0
  __wl_bound_00056_lower: + 1 s[W3,C6] >= 0
  __wl_bound_00057_lower: + 1 s[W3,C7] >= 0
  __wl_bound_00058_lower: + 1 s[W3,C8] >= 0
  __wl_bound_00059_lower: + 1 s[W3,C9] >= 0
  __wl_bound_00060_lower: + 1 s[W3,C10] >= 0
  __wl_bound_00061_lower: + 1 s[W3,C11] >= 0
  __wl_bound_00062_lower: + 1 s[W3,C12] >= 0
  __wl_bound_00063_lower: + 1 s[W3,C13] >= 0
  __wl_bound_00064_lower: + 1 s[W3,C14] >= 0
  __wl_bound_00065_lower: + 1 s[W3,C15] >= 0
  __wl_bound_00066_lower: + 1 s[W3,C16] >= 0
  __wl_bound_00067_lower: + 1 s[W3,C17] >= 0
  __wl_bound_00068_lower: + 1 s[W3,C18] >= 0
  __wl_bound_00069_lower: + 1 s[W3,C19] >= 0
  __wl_bound_00070_lower: + 1 s[W3,C20] >= 0
  __wl_bound_00071_lower: + 1 s[W4,C1] >= 0
  __wl_bound_00072_lower: + 1 s[W4,C2] >= 0
  __wl_bound_00073_lower: + 1 s[W4,C3] >= 0
  __wl_bound_00074_lower: + 1 s[W4,C4] >= 0
  __wl_bound_00075_lower: + 1 s[W4,C5] >= 0
  __wl_bound_00076_lower: + 1 s[W4,C6] >= 0
  __wl_bound_00077_lower: + 1 s[W4,C7] >= 0
  __wl_bound_00078_lower: + 1 s[W4,C8] >= 0
  __wl_bound_00079_lower: + 1 s[W4,C9] >= 0
  __wl_bound_00080_lower: + 1 s[W4,C10] >= 0
  __wl_bound_00081_lower: + 1 s[W4,C11] >= 0
  __wl_bound_00082_lower: + 1 s[W4,C12] >= 0
  __wl_bound_00083_lower: + 1 s[W4,C13] >= 0
  __wl_bound_00084_lower: + 1 s[W4,C14] >= 0
  __wl_bound_00085_lower: + 1 s[W4,C15] >= 0
  __wl_bound_00086_lower: + 1 s[W4,C16] >= 0
  __wl_bound_00087_lower: + 1 s[W4,C17] >= 0
  __wl_bound_00088_lower: + 1 s[W4,C18] >= 0
  __wl_bound_00089_lower: + 1 s[W4,C19] >= 0
  __wl_bound_00090_lower: + 1 s[W4,C20] >= 0
  __wl_bound_00091_lower: + 1 s[W5,C1] >= 0
  __wl_bound_00092_lower: + 1 s[W5,C2] >= 0
  __wl_bound_00093_lower: + 1 s[W5,C3] >= 0
  __wl_bound_00094_lower: + 1 s[W5,C4] >= 0
  __wl_bound_00095_lower: + 1 s[W5,C5] >= 0
  __wl_bound_00096_lower: + 1 s[W5,C6] >= 0
  __wl_bound_00097_lower: + 1 s[W5,C7] >= 0
  __wl_bound_00098_lower: + 1 s[W5,C8] >= 0
  __wl_bound_00099_lower: + 1 s[W5,C9] >= 0
  __wl_bound_00100_lower: + 1 s[W5,C10] >= 0
  __wl_bound_00101_lower: + 1 s[W5,C11] >= 0
  __wl_bound_00102_lower: + 1 s[W5,C12] >= 0
  __wl_bound_00103_lower: + 1 s[W5,C13] >= 0
  __wl_bound_00104_lower: + 1 s[W5,C14] >= 0
  __wl_bound_00105_lower: + 1 s[W5,C15] >= 0
  __wl_bound_00106_lower: + 1 s[W5,C16] >= 0
  __wl_bound_00107_lower: + 1 s[W5,C17] >= 0
  __wl_bound_00108_lower: + 1 s[W5,C18] >= 0
  __wl_bound_00109_lower: + 1 s[W5,C19] >= 0
  __wl_bound_00110_lower: + 1 s[W5,C20] >= 0
  __wl_bound_00111_lower: + 1 s[W6,C1] >= 0
  __wl_bound_00112_lower: + 1 s[W6,C2] >= 0
  __wl_bound_00113_lower: + 1 s[W6,C3] >= 0
  __wl_bound_00114_lower: + 1 s[W6,C4] >= 0
  __wl_bound_00115_lower: + 1 s[W6,C5] >= 0
  __wl_bound_00116_lower: + 1 s[W6,C6] >= 0
  __wl_bound_00117_lower: + 1 s[W6,C7] >= 0
  __wl_bound_00118_lower: + 1 s[W6,C8] >= 0
  __wl_bound_00119_lower: + 1 s[W6,C9] >= 0
  __wl_bound_00120_lower: + 1 s[W6,C10] >= 0
  __wl_bound_00121_lower: + 1 s[W6,C11] >= 0
  __wl_bound_00122_lower: + 1 s[W6,C12] >= 0
  __wl_bound_00123_lower: + 1 s[W6,C13] >= 0
  __wl_bound_00124_lower: + 1 s[W6,C14] >= 0
  __wl_bound_00125_lower: + 1 s[W6,C15] >= 0
  __wl_bound_00126_lower: + 1 s[W6,C16] >= 0
  __wl_bound_00127_lower: + 1 s[W6,C17] >= 0
  __wl_bound_00128_lower: + 1 s[W6,C18] >= 0
  __wl_bound_00129_lower: + 1 s[W6,C19] >= 0
  __wl_bound_00130_lower: + 1 s[W6,C20] >= 0
  __wl_bound_00131_lower: + 1 s[W7,C1] >= 0
  __wl_bound_00132_lower: + 1 s[W7,C2] >= 0
  __wl_bound_00133_lower: + 1 s[W7,C3] >= 0
  __wl_bound_00134_lower: + 1 s[W7,C4] >= 0
  __wl_bound_00135_lower: + 1 s[W7,C5] >= 0
  __wl_bound_00136_lower: + 1 s[W7,C6] >= 0
  __wl_bound_00137_lower: + 1 s[W7,C7] >= 0
  __wl_bound_00138_lower: + 1 s[W7,C8] >= 0
  __wl_bound_00139_lower: + 1 s[W7,C9] >= 0
  __wl_bound_00140_lower: + 1 s[W7,C10] >= 0
  __wl_bound_00141_lower: + 1 s[W7,C11] >= 0
  __wl_bound_00142_lower: + 1 s[W7,C12] >= 0
  __wl_bound_00143_lower: + 1 s[W7,C13] >= 0
  __wl_bound_00144_lower: + 1 s[W7,C14] >= 0
  __wl_bound_00145_lower: + 1 s[W7,C15] >= 0
  __wl_bound_00146_lower: + 1 s[W7,C16] >= 0
  __wl_bound_00147_lower: + 1 s[W7,C17] >= 0
  __wl_bound_00148_lower: + 1 s[W7,C18] >= 0
  __wl_bound_00149_lower: + 1 s[W7,C19] >= 0
  __wl_bound_00150_lower: + 1 s[W7,C20] >= 0
  __wl_bound_00151_lower: + 1 s[W8,C1] >= 0
  __wl_bound_00152_lower: + 1 s[W8,C2] >= 0
  __wl_bound_00153_lower: + 1 s[W8,C3] >= 0
  __wl_bound_00154_lower: + 1 s[W8,C4] >= 0
  __wl_bound_00155_lower: + 1 s[W8,C5] >= 0
  __wl_bound_00156_lower: + 1 s[W8,C6] >= 0
  __wl_bound_00157_lower: + 1 s[W8,C7] >= 0
  __wl_bound_00158_lower: + 1 s[W8,C8] >= 0
  __wl_bound_00159_lower: + 1 s[W8,C9] >= 0
  __wl_bound_00160_lower: + 1 s[W8,C10] >= 0
  __wl_bound_00161_lower: + 1 s[W8,C11] >= 0
  __wl_bound_00162_lower: + 1 s[W8,C12] >= 0
  __wl_bound_00163_lower: + 1 s[W8,C13] >= 0
  __wl_bound_00164_lower: + 1 s[W8,C14] >= 0
  __wl_bound_00165_lower: + 1 s[W8,C15] >= 0
  __wl_bound_00166_lower: + 1 s[W8,C16] >= 0
  __wl_bound_00167_lower: + 1 s[W8,C17] >= 0
  __wl_bound_00168_lower: + 1 s[W8,C18] >= 0
  __wl_bound_00169_lower: + 1 s[W8,C19] >= 0
  __wl_bound_00170_lower: + 1 s[W8,C20] >= 0
  __wl_bound_00171_lower: + 1 s[W9,C1] >= 0
  __wl_bound_00172_lower: + 1 s[W9,C2] >= 0
  __wl_bound_00173_lower: + 1 s[W9,C3] >= 0
  __wl_bound_00174_lower: + 1 s[W9,C4] >= 0
  __wl_bound_00175_lower: + 1 s[W9,C5] >= 0
  __wl_bound_00176_lower: + 1 s[W9,C6] >= 0
  __wl_bound_00177_lower: + 1 s[W9,C7] >= 0
  __wl_bound_00178_lower: + 1 s[W9,C8] >= 0
  __wl_bound_00179_lower: + 1 s[W9,C9] >= 0
  __wl_bound_00180_lower: + 1 s[W9,C10] >= 0
  __wl_bound_00181_lower: + 1 s[W9,C11] >= 0
  __wl_bound_00182_lower: + 1 s[W9,C12] >= 0
  __wl_bound_00183_lower: + 1 s[W9,C13] >= 0
  __wl_bound_00184_lower: + 1 s[W9,C14] >= 0
  __wl_bound_00185_lower: + 1 s[W9,C15] >= 0
  __wl_bound_00186_lower: + 1 s[W9,C16] >= 0
  __wl_bound_00187_lower: + 1 s[W9,C17] >= 0
  __wl_bound_00188_lower: + 1 s[W9,C18] >= 0
  __wl_bound_00189_lower: + 1 s[W9,C19] >= 0
  __wl_bound_00190_lower: + 1 s[W9,C20] >= 0
  __wl_bound_00191_lower: + 1 s[W10,C1] >= 0
  __wl_bound_00192_lower: + 1 s[W10,C2] >= 0
  __wl_bound_00193_lower: + 1 s[W10,C3] >= 0
  __wl_bound_00194_lower: + 1 s[W10,C4] >= 0
  __wl_bound_00195_lower: + 1 s[W10,C5] >= 0
  __wl_bound_00196_lower: + 1 s[W10,C6] >= 0
  __wl_bound_00197_lower: + 1 s[W10,C7] >= 0
  __wl_bound_00198_lower: + 1 s[W10,C8] >= 0
  __wl_bound_00199_lower: + 1 s[W10,C9] >= 0
  __wl_bound_00200_lower: + 1 s[W10,C10] >= 0
  __wl_bound_00201_lower: + 1 s[W10,C11] >= 0
  __wl_bound_00202_lower: + 1 s[W10,C12] >= 0
  __wl_bound_00203_lower: + 1 s[W10,C13] >= 0
  __wl_bound_00204_lower: + 1 s[W10,C14] >= 0
  __wl_bound_00205_lower: + 1 s[W10,C15] >= 0
  __wl_bound_00206_lower: + 1 s[W10,C16] >= 0
  __wl_bound_00207_lower: + 1 s[W10,C17] >= 0
  __wl_bound_00208_lower: + 1 s[W10,C18] >= 0
  __wl_bound_00209_lower: + 1 s[W10,C19] >= 0
  __wl_bound_00210_lower: + 1 s[W10,C20] >= 0
Bounds
  s[W1,C1] free
  s[W1,C2] free
  s[W1,C3] free
  s[W1,C4] free
  s[W1,C5] free
  s[W1,C6] free
  s[W1,C7] free
  s[W1,C8] free
  s[W1,C9] free
  s[W1,C10] free
  s[W1,C11] free
  s[W1,C12] free
  s[W1,C13] free
  s[W1,C14] free
  s[W1,C15] free
  s[W1,C16] free
  s[W1,C17] free
  s[W1,C18] free
  s[W1,C19] free
  s[W1,C20] free
  s[W2,C1] free
  s[W2,C2] free
  s[W2,C3] free
  s[W2,C4] free
  s[W2,C5] free
  s[W2,C6] free
  s[W2,C7] free
  s[W2,C8] free
  s[W2,C9] free
  s[W2,C10] free
  s[W2,C11] free
  s[W2,C12] free
  s[W2,C13] free
  s[W2,C14] free
  s[W2,C15] free
  s[W2,C16] free
  s[W2,C17] free
  s[W2,C18] free
  s[W2,C19] free
  s[W2,C20] free
  s[W3,C1] free
  s[W3,C2] free
  s[W3,C3] free
  s[W3,C4] free
  s[W3,C5] free
  s[W3,C6] free
  s[W3,C7] free
  s[W3,C8] free
  s[W3,C9] free
  s[W3,C10] free
  s[W3,C11] free
  s[W3,C12] free
  s[W3,C13] free
  s[W3,C14] free
  s[W3,C15] free
  s[W3,C16] free
  s[W3,C17] free
  s[W3,C18] free
  s[W3,C19] free
  s[W3,C20] free
  s[W4,C1] free
  s[W4,C2] free
  s[W4,C3] free
  s[W4,C4] free
  s[W4,C5] free
  s[W4,C6] free
  s[W4,C7] free
  s[W4,C8] free
  s[W4,C9] free
  s[W4,C10] free
  s[W4,C11] free
  s[W4,C12] free
  s[W4,C13] free
  s[W4,C14] free
  s[W4,C15] free
  s[W4,C16] free
  s[W4,C17] free
  s[W4,C18] free
  s[W4,C19] free
  s[W4,C20] free
  s[W5,C1] free
  s[W5,C2] free
  s[W5,C3] free
  s[W5,C4] free
  s[W5,C5] free
  s[W5,C6] free
  s[W5,C7] free
  s[W5,C8] free
  s[W5,C9] free
  s[W5,C10] free
  s[W5,C11] free
  s[W5,C12] free
  s[W5,C13] free
  s[W5,C14] free
  s[W5,C15] free
  s[W5,C16] free
  s[W5,C17] free
  s[W5,C18] free
  s[W5,C19] free
  s[W5,C20] free
  s[W6,C1] free
  s[W6,C2] free
  s[W6,C3] free
  s[W6,C4] free
  s[W6,C5] free
  s[W6,C6] free
  s[W6,C7] free
  s[W6,C8] free
  s[W6,C9] free
  s[W6,C10] free
  s[W6,C11] free
  s[W6,C12] free
  s[W6,C13] free
  s[W6,C14] free
  s[W6,C15] free
  s[W6,C16] free
  s[W6,C17] free
  s[W6,C18] free
  s[W6,C19] free
  s[W6,C20] free
  s[W7,C1] free
  s[W7,C2] free
  s[W7,C3] free
  s[W7,C4] free
  s[W7,C5] free
  s[W7,C6] free
  s[W7,C7] free
  s[W7,C8] free
  s[W7,C9] free
  s[W7,C10] free
  s[W7,C11] free
  s[W7,C12] free
  s[W7,C13] free
  s[W7,C14] free
  s[W7,C15] free
  s[W7,C16] free
  s[W7,C17] free
  s[W7,C18] free
  s[W7,C19] free
  s[W7,C20] free
  s[W8,C1] free
  s[W8,C2] free
  s[W8,C3] free
  s[W8,C4] free
  s[W8,C5] free
  s[W8,C6] free
  s[W8,C7] free
  s[W8,C8] free
  s[W8,C9] free
  s[W8,C10] free
  s[W8,C11] free
  s[W8,C12] free
  s[W8,C13] free
  s[W8,C14] free
  s[W8,C15] free
  s[W8,C16] free
  s[W8,C17] free
  s[W8,C18] free
  s[W8,C19] free
  s[W8,C20] free
  s[W9,C1] free
  s[W9,C2] free
  s[W9,C3] free
  s[W9,C4] free
  s[W9,C5] free
  s[W9,C6] free
  s[W9,C7] free
  s[W9,C8] free
  s[W9,C9] free
  s[W9,C10] free
  s[W9,C11] free
  s[W9,C12] free
  s[W9,C13] free
  s[W9,C14] free
  s[W9,C15] free
  s[W9,C16] free
  s[W9,C17] free
  s[W9,C18] free
  s[W9,C19] free
  s[W9,C20] free
  s[W10,C1] free
  s[W10,C2] free
  s[W10,C3] free
  s[W10,C4] free
  s[W10,C5] free
  s[W10,C6] free
  s[W10,C7] free
  s[W10,C8] free
  s[W10,C9] free
  s[W10,C10] free
  s[W10,C11] free
  s[W10,C12] free
  s[W10,C13] free
  s[W10,C14] free
  s[W10,C15] free
  s[W10,C16] free
  s[W10,C17] free
  s[W10,C18] free
  s[W10,C19] free
  s[W10,C20] free
Binaries
  y[W1] y[W2] y[W3] y[W4] y[W5] y[W6] y[W7] y[W8] y[W9] y[W10]
End
