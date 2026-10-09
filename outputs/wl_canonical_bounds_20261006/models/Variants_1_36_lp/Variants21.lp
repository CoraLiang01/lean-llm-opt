\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 120 z_A + 150 z_B + 180 z_C + 160 z_D + 210 z_E + 140 z_F + 170 z_G + 200 z_H + 260 z_I
Subject To
  precedence_A_C: + 1 z_A + 1 s_C - 1 s_A >= 4
  precedence_A_D: + 1 z_A - 1 s_A + 1 s_D >= 4
  precedence_B_E: + 1 z_B + 1 s_E - 1 s_B >= 7
  precedence_C_F: + 1 z_C - 1 s_C + 1 s_F >= 6
  precedence_D_F: + 1 z_D - 1 s_D + 1 s_F >= 5
  precedence_D_G: + 1 z_D - 1 s_D + 1 s_G >= 5
  precedence_E_G: + 1 z_E - 1 s_E + 1 s_G >= 4
  precedence_F_H: + 1 z_F - 1 s_F + 1 s_H >= 7
  precedence_G_H: + 1 z_G - 1 s_G + 1 s_H >= 6
  precedence_H_I: + 1 z_H - 1 s_H + 1 s_I >= 5
  completion_A: + 1 z_A - 1 s_A + 1 T >= 4
  completion_B: + 1 z_B - 1 s_B + 1 T >= 7
  completion_C: + 1 z_C - 1 s_C + 1 T >= 6
  completion_D: + 1 z_D - 1 s_D + 1 T >= 5
  completion_E: + 1 z_E - 1 s_E + 1 T >= 4
  completion_F: + 1 z_F - 1 s_F + 1 T >= 7
  completion_G: + 1 z_G - 1 s_G + 1 T >= 6
  completion_H: + 1 z_H - 1 s_H + 1 T >= 5
  completion_I: + 1 z_I - 1 s_I + 1 T >= 3
  __wl_bound_00001_lower: + 1 z_A >= 0
  __wl_bound_00001_upper: + 1 z_A <= 1
  __wl_bound_00002_lower: + 1 z_B >= 0
  __wl_bound_00002_upper: + 1 z_B <= 2
  __wl_bound_00003_lower: + 1 z_C >= 0
  __wl_bound_00003_upper: + 1 z_C <= 2
  __wl_bound_00004_lower: + 1 z_D >= 0
  __wl_bound_00004_upper: + 1 z_D <= 2
  __wl_bound_00005_lower: + 1 z_E >= 0
  __wl_bound_00005_upper: + 1 z_E <= 2
  __wl_bound_00006_lower: + 1 z_F >= 0
  __wl_bound_00006_upper: + 1 z_F <= 2
  __wl_bound_00007_lower: + 1 z_G >= 0
  __wl_bound_00007_upper: + 1 z_G <= 2
  __wl_bound_00008_lower: + 1 z_H >= 0
  __wl_bound_00008_upper: + 1 z_H <= 2
  __wl_bound_00009_lower: + 1 z_I >= 0
  __wl_bound_00009_upper: + 1 z_I <= 1
  __wl_bound_00010_lower: + 1 s_C >= 0
  __wl_bound_00011_lower: + 1 s_A >= 0
  __wl_bound_00012_lower: + 1 s_D >= 0
  __wl_bound_00013_lower: + 1 s_E >= 0
  __wl_bound_00014_lower: + 1 s_B >= 0
  __wl_bound_00015_lower: + 1 s_F >= 0
  __wl_bound_00016_lower: + 1 s_G >= 0
  __wl_bound_00017_lower: + 1 s_H >= 0
  __wl_bound_00018_lower: + 1 s_I >= 0
  __wl_bound_00019_lower: + 1 T >= 0
  __wl_bound_00019_upper: + 1 T <= 22
Bounds
  z_A free
  z_B free
  z_C free
  z_D free
  z_E free
  z_F free
  z_G free
  z_H free
  z_I free
  s_C free
  s_A free
  s_D free
  s_E free
  s_B free
  s_F free
  s_G free
  s_H free
  s_I free
  T free
Generals
  z_A z_B z_C z_D z_E z_F z_G z_H z_I
End
