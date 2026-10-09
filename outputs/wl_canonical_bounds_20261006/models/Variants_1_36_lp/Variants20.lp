\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 8 y_S1 + 12 y_S2 + 11 y_S3 + 10 y_S4 + 13 y_S5 + 9 y_S6 + 14 y_S7 + 7 y_S8 + 15 y_S9
Subject To
  cover_Q1: + 1 y_S1 + 1 y_S7 >= 1
  cover_Q2: + 1 y_S1 + 1 y_S2 + 1 y_S9 >= 1
  cover_Q3: + 1 y_S2 + 1 y_S8 >= 1
  cover_Q4: + 1 y_S1 + 1 y_S3 >= 1
  cover_Q5: + 1 y_S2 + 1 y_S4 + 1 y_S8 >= 1
  cover_Q6: + 1 y_S3 + 1 y_S5 + 1 y_S9 >= 1
  cover_Q7: + 1 y_S3 + 1 y_S6 >= 1
  cover_Q8: + 1 y_S4 + 1 y_S5 >= 1
  cover_Q9: + 1 y_S5 + 1 y_S7 >= 1
  cover_Q10: + 1 y_S6 + 1 y_S7 + 1 y_S9 >= 1
  cover_Q11: + 1 y_S6 + 1 y_S8 >= 1
  __wl_bound_00001_lower: + 1 y_S1 >= 0
  __wl_bound_00001_upper: + 1 y_S1 <= 1
  __wl_bound_00002_lower: + 1 y_S2 >= 0
  __wl_bound_00002_upper: + 1 y_S2 <= 1
  __wl_bound_00003_lower: + 1 y_S3 >= 0
  __wl_bound_00003_upper: + 1 y_S3 <= 1
  __wl_bound_00004_lower: + 1 y_S4 >= 0
  __wl_bound_00004_upper: + 1 y_S4 <= 1
  __wl_bound_00005_lower: + 1 y_S5 >= 0
  __wl_bound_00005_upper: + 1 y_S5 <= 1
  __wl_bound_00006_lower: + 1 y_S6 >= 0
  __wl_bound_00006_upper: + 1 y_S6 <= 1
  __wl_bound_00007_lower: + 1 y_S7 >= 0
  __wl_bound_00007_upper: + 1 y_S7 <= 1
  __wl_bound_00008_lower: + 1 y_S8 >= 0
  __wl_bound_00008_upper: + 1 y_S8 <= 1
  __wl_bound_00009_lower: + 1 y_S9 >= 0
  __wl_bound_00009_upper: + 1 y_S9 <= 1
Bounds
Binaries
  y_S1 y_S2 y_S3 y_S4 y_S5 y_S6 y_S7 y_S8 y_S9
End
