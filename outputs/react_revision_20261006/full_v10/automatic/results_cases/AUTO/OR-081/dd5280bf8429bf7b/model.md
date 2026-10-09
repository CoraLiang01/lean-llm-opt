Mathematical Model

Sets:
F = {all foods listed in cost.csv, identified by Food}

Parameters (from cost.csv, table_id: file_0_view_0, columns as named):
cal_f = Calories per serving of food f, for all f in F
prot_f = Protein(g) per serving of food f, for all f in F
fat_f = Fat(g) per serving of food f, for all f in F
vitc_f = VitaminC(mg) per serving of food f, for all f in F
cost_f = Cost (USD) per serving of food f, for all f in F

Decision Variables:
x_f ≥ 0 : number of servings of food f to include in the meal plan (continuous, may be fractional), for all f in F

Objective:
minimize   ∑_{f∈F} cost_f x_f

Subject to:
∑_{f∈F} cal_f x_f ≥ 2000
∑_{f∈F} prot_f x_f ≥ 50
∑_{f∈F} vitc_f x_f ≥ 60
∑_{f∈F} fat_f x_f ≤ 70
x_f ≥ 0   for all f ∈ F

Data Mapping:
All parameters cal_f, prot_f, fat_f, vitc_f, cost_f are mapped directly from the corresponding columns in cost.csv (table_id: file_0_view_0), indexed by Food. The set F is the set of all Food entries in cost.csv.