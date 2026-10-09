\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Maximize
  obj: - 0.5 x[Red,I] + 1 x[Red,II] + 2.5 x[Red,III] - 1 x[Yellow,I] + 0.5 x[Yellow,II] + 2 x[Yellow,III] - 1.2 x[Blue,I] + 0.29999999999999999 x[Blue,II] + 1.8 x[Blue,III]
Subject To
  supply_I: + 1 x[Red,I] + 1 x[Yellow,I] + 1 x[Blue,I] <= 1500
  supply_II: + 1 x[Red,II] + 1 x[Yellow,II] + 1 x[Blue,II] <= 2000
  supply_III: + 1 x[Red,III] + 1 x[Yellow,III] + 1 x[Blue,III] <= 1000
  Red_I_upper_ratio: + 0.90000000000000002 x[Red,I] - 0.10000000000000001 x[Red,II] - 0.10000000000000001 x[Red,III] <= 0
  Red_II_lower_ratio: - 0.5 x[Red,I] + 0.5 x[Red,II] - 0.5 x[Red,III] >= 0
  Yellow_III_upper_ratio: - 0.69999999999999996 x[Yellow,I] - 0.69999999999999996 x[Yellow,II] + 0.29999999999999999 x[Yellow,III] <= 0
  Yellow_I_lower_ratio: + 0.80000000000000004 x[Yellow,I] - 0.20000000000000001 x[Yellow,II] - 0.20000000000000001 x[Yellow,III] >= 0
  Blue_I_upper_ratio: + 0.5 x[Blue,I] - 0.5 x[Blue,II] - 0.5 x[Blue,III] <= 0
  Blue_II_lower_ratio: - 0.10000000000000001 x[Blue,I] + 0.90000000000000002 x[Blue,II] - 0.10000000000000001 x[Blue,III] >= 0
  min_production_Red: + 1 x[Red,I] + 1 x[Red,II] + 1 x[Red,III] >= 2000
  __wl_bound_00001_lower: + 1 x[Red,I] >= 0
  __wl_bound_00002_lower: + 1 x[Red,II] >= 0
  __wl_bound_00003_lower: + 1 x[Red,III] >= 0
  __wl_bound_00004_lower: + 1 x[Yellow,I] >= 0
  __wl_bound_00005_lower: + 1 x[Yellow,II] >= 0
  __wl_bound_00006_lower: + 1 x[Yellow,III] >= 0
  __wl_bound_00007_lower: + 1 x[Blue,I] >= 0
  __wl_bound_00008_lower: + 1 x[Blue,II] >= 0
  __wl_bound_00009_lower: + 1 x[Blue,III] >= 0
Bounds
  x[Red,I] free
  x[Red,II] free
  x[Red,III] free
  x[Yellow,I] free
  x[Yellow,II] free
  x[Yellow,III] free
  x[Blue,I] free
  x[Blue,II] free
  x[Blue,III] free
End
