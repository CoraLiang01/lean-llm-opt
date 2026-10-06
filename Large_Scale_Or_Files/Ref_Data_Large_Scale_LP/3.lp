\ Model Original_RAG_NRM
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xb8e99a53d4098aac
Maximize
  9097 x[Nike_x_Olivia_Kim_Air_Footscape]
   + 9995 x[Nike_x_Olivia_Kim_Air_Force_1_'07]
   + 11197 x[Nike_x_Olivia_Kim_Air_Max_98]
   + 11197 x[Nike_x_Olivia_Kim_Air_Mowabb]
Subject To
 inventory[Nike_x_Olivia_Kim_Air_Footscape]:
   x[Nike_x_Olivia_Kim_Air_Footscape] <= 240
 inventory[Nike_x_Olivia_Kim_Air_Force_1_'07]:
   x[Nike_x_Olivia_Kim_Air_Force_1_'07] <= 281
 inventory[Nike_x_Olivia_Kim_Air_Max_98]: x[Nike_x_Olivia_Kim_Air_Max_98]
   <= 322
 inventory[Nike_x_Olivia_Kim_Air_Mowabb]: x[Nike_x_Olivia_Kim_Air_Mowabb]
   <= 97
 demand[Nike_x_Olivia_Kim_Air_Footscape]:
   x[Nike_x_Olivia_Kim_Air_Footscape] <= 26
 demand[Nike_x_Olivia_Kim_Air_Force_1_'07]:
   x[Nike_x_Olivia_Kim_Air_Force_1_'07] <= 53
 demand[Nike_x_Olivia_Kim_Air_Max_98]: x[Nike_x_Olivia_Kim_Air_Max_98]
   <= 50
 demand[Nike_x_Olivia_Kim_Air_Mowabb]: x[Nike_x_Olivia_Kim_Air_Mowabb]
   <= 17
 dispatch_capacity: x[Nike_x_Olivia_Kim_Air_Footscape]
   + x[Nike_x_Olivia_Kim_Air_Force_1_'07] + x[Nike_x_Olivia_Kim_Air_Max_98]
   + x[Nike_x_Olivia_Kim_Air_Mowabb] <= 100
Bounds
 x[Nike_x_Olivia_Kim_Air_Footscape] >= 1
 x[Nike_x_Olivia_Kim_Air_Force_1_'07] >= 1
 x[Nike_x_Olivia_Kim_Air_Max_98] >= 1
 x[Nike_x_Olivia_Kim_Air_Mowabb] >= 1
Generals
 x[Nike_x_Olivia_Kim_Air_Footscape] x[Nike_x_Olivia_Kim_Air_Force_1_'07]
 x[Nike_x_Olivia_Kim_Air_Max_98] x[Nike_x_Olivia_Kim_Air_Mowabb]
End
