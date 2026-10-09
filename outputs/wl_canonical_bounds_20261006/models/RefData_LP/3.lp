\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Maximize
  obj: + 9097 x[Nike_x_Olivia_Kim_Air_Footscape] + 9995 x[Nike_x_Olivia_Kim_Air_Force_1_'07] + 11197 x[Nike_x_Olivia_Kim_Air_Max_98] + 11197 x[Nike_x_Olivia_Kim_Air_Mowabb]
Subject To
  dispatch_capacity: + 1 x[Nike_x_Olivia_Kim_Air_Footscape] + 1 x[Nike_x_Olivia_Kim_Air_Force_1_'07] + 1 x[Nike_x_Olivia_Kim_Air_Max_98] + 1 x[Nike_x_Olivia_Kim_Air_Mowabb] <= 100
  __wl_bound_00001_lower: + 1 x[Nike_x_Olivia_Kim_Air_Footscape] >= 1
  __wl_bound_00001_upper: + 1 x[Nike_x_Olivia_Kim_Air_Footscape] <= 26
  __wl_bound_00002_lower: + 1 x[Nike_x_Olivia_Kim_Air_Force_1_'07] >= 1
  __wl_bound_00002_upper: + 1 x[Nike_x_Olivia_Kim_Air_Force_1_'07] <= 53
  __wl_bound_00003_lower: + 1 x[Nike_x_Olivia_Kim_Air_Max_98] >= 1
  __wl_bound_00003_upper: + 1 x[Nike_x_Olivia_Kim_Air_Max_98] <= 50
  __wl_bound_00004_lower: + 1 x[Nike_x_Olivia_Kim_Air_Mowabb] >= 1
  __wl_bound_00004_upper: + 1 x[Nike_x_Olivia_Kim_Air_Mowabb] <= 17
Bounds
  x[Nike_x_Olivia_Kim_Air_Footscape] free
  x[Nike_x_Olivia_Kim_Air_Force_1_'07] free
  x[Nike_x_Olivia_Kim_Air_Max_98] free
  x[Nike_x_Olivia_Kim_Air_Mowabb] free
Generals
  x[Nike_x_Olivia_Kim_Air_Footscape] x[Nike_x_Olivia_Kim_Air_Force_1_'07] x[Nike_x_Olivia_Kim_Air_Max_98] x[Nike_x_Olivia_Kim_Air_Mowabb]
End
