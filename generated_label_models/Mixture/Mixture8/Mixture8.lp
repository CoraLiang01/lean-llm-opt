\ Model Mixture8_LabelModel
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - 0.5 x[Red,I] + x[Red,II] + 2.5 x[Red,III] - x[Yellow,I]
   + 0.5 x[Yellow,II] + 2 x[Yellow,III] - 1.2 x[Blue,I] + 0.3 x[Blue,II]
   + 1.8 x[Blue,III]
Subject To
 supply_I: x[Red,I] + x[Yellow,I] + x[Blue,I] <= 1500
 supply_II: x[Red,II] + x[Yellow,II] + x[Blue,II] <= 2000
 supply_III: x[Red,III] + x[Yellow,III] + x[Blue,III] <= 1000
 Red_I_upper_ratio: 0.9 x[Red,I] - 0.1 x[Red,II] - 0.1 x[Red,III] <= 0
 Red_II_lower_ratio: - 0.5 x[Red,I] + 0.5 x[Red,II] - 0.5 x[Red,III] >= 0
 Yellow_III_upper_ratio: - 0.7 x[Yellow,I] - 0.7 x[Yellow,II]
   + 0.3 x[Yellow,III] <= 0
 Yellow_I_lower_ratio: 0.8 x[Yellow,I] - 0.2 x[Yellow,II]
   - 0.2 x[Yellow,III] >= 0
 Blue_I_upper_ratio: 0.5 x[Blue,I] - 0.5 x[Blue,II] - 0.5 x[Blue,III] <= 0
 Blue_II_lower_ratio: - 0.1 x[Blue,I] + 0.9 x[Blue,II] - 0.1 x[Blue,III]
   >= 0
 min_production_Red: x[Red,I] + x[Red,II] + x[Red,III] >= 2000
Bounds
End

