\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 180 z_A + 150 z_B + 130 z_C + 210 z_D + 160 z_E + 190 z_F + 140 z_G + 220 z_H + 170 z_I + 260 z_J
Subject To
  precedence_A_C: + 1 z_A + 1 s_C - 1 s_A >= 5
  precedence_A_D: + 1 z_A - 1 s_A + 1 s_D >= 5
  precedence_B_E: + 1 z_B + 1 s_E - 1 s_B >= 6
  precedence_C_F: + 1 z_C - 1 s_C + 1 s_F >= 7
  precedence_D_F: + 1 z_D - 1 s_D + 1 s_F >= 4
  precedence_D_G: + 1 z_D - 1 s_D + 1 s_G >= 4
  precedence_E_G: + 1 z_E - 1 s_E + 1 s_G >= 5
  precedence_F_H: + 1 z_F - 1 s_F + 1 s_H >= 6
  precedence_G_I: + 1 z_G - 1 s_G + 1 s_I >= 7
  precedence_H_J: + 1 z_H - 1 s_H + 1 s_J >= 4
  precedence_I_J: + 1 z_I - 1 s_I + 1 s_J >= 5
  completion_A: + 1 z_A - 1 s_A + 1 T >= 5
  completion_B: + 1 z_B - 1 s_B + 1 T >= 6
  completion_C: + 1 z_C - 1 s_C + 1 T >= 7
  completion_D: + 1 z_D - 1 s_D + 1 T >= 4
  completion_E: + 1 z_E - 1 s_E + 1 T >= 5
  completion_F: + 1 z_F - 1 s_F + 1 T >= 6
  completion_G: + 1 z_G - 1 s_G + 1 T >= 7
  completion_H: + 1 z_H - 1 s_H + 1 T >= 4
  completion_I: + 1 z_I - 1 s_I + 1 T >= 5
  completion_J: + 1 z_J - 1 s_J + 1 T >= 3
  __wl_bound_00001_lower: + 1 z_A >= 0
  __wl_bound_00001_upper: + 1 z_A <= 2
  __wl_bound_00002_lower: + 1 z_B >= 0
  __wl_bound_00002_upper: + 1 z_B <= 2
  __wl_bound_00003_lower: + 1 z_C >= 0
  __wl_bound_00003_upper: + 1 z_C <= 3
  __wl_bound_00004_lower: + 1 z_D >= 0
  __wl_bound_00004_upper: + 1 z_D <= 1
  __wl_bound_00005_lower: + 1 z_E >= 0
  __wl_bound_00005_upper: + 1 z_E <= 2
  __wl_bound_00006_lower: + 1 z_F >= 0
  __wl_bound_00006_upper: + 1 z_F <= 2
  __wl_bound_00007_lower: + 1 z_G >= 0
  __wl_bound_00007_upper: + 1 z_G <= 2
  __wl_bound_00008_lower: + 1 z_H >= 0
  __wl_bound_00008_upper: + 1 z_H <= 2
  __wl_bound_00009_lower: + 1 z_I >= 0
  __wl_bound_00009_upper: + 1 z_I <= 2
  __wl_bound_00010_lower: + 1 z_J >= 0
  __wl_bound_00010_upper: + 1 z_J <= 1
  __wl_bound_00011_lower: + 1 s_C >= 0
  __wl_bound_00012_lower: + 1 s_A >= 0
  __wl_bound_00013_lower: + 1 s_D >= 0
  __wl_bound_00014_lower: + 1 s_E >= 0
  __wl_bound_00015_lower: + 1 s_B >= 0
  __wl_bound_00016_lower: + 1 s_F >= 0
  __wl_bound_00017_lower: + 1 s_G >= 0
  __wl_bound_00018_lower: + 1 s_H >= 0
  __wl_bound_00019_lower: + 1 s_I >= 0
  __wl_bound_00020_lower: + 1 s_J >= 0
  __wl_bound_00021_lower: + 1 T >= 0
  __wl_bound_00021_upper: + 1 T <= 23
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
  z_J free
  s_C free
  s_A free
  s_D free
  s_E free
  s_B free
  s_F free
  s_G free
  s_H free
  s_I free
  s_J free
  T free
Generals
  z_A z_B z_C z_D z_E z_F z_G z_H z_I z_J
End
