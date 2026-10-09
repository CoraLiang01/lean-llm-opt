Mathematical Model

Sets:
F = set of all foods in cost.csv (indexed by f)

Parameters (from cost.csv, for each f ∈ F):
cal_f = Calories per serving of food f
prot_f = Protein (g) per serving of food f
fat_f = Fat (g) per serving of food f
vitc_f = VitaminC (mg) per serving of food f
cost_f = Cost (USD) per serving of food f

Decision Variables:
x_f ≥ 0 : number of servings of food f (can be fractional)

Objective:
Minimize total cost:
minimize   ∑_{f∈F} cost_f * x_f

Subject to:
Calorie requirement:
∑_{f∈F} cal_f * x_f ≥ 2000

Protein requirement:
∑_{f∈F} prot_f * x_f ≥ 50

Vitamin C requirement:
∑_{f∈F} vitc_f * x_f ≥ 60

Fat upper bound:
∑_{f∈F} fat_f * x_f ≤ 70

Non-negativity:
x_f ≥ 0   for all f ∈ F

Data Mapping

Set F and all parameters cal_f, prot_f, fat_f, vitc_f, cost_f are taken from cost.csv (table_id: file_0_view_0), with columns:
- Food: f
- Calories: cal_f
- Protein(g): prot_f
- Fat(g): fat_f
- VitaminC(mg): vitc_f
- Cost: cost_f

All 120 foods in cost.csv are included in F. All constraints and coefficients are mapped directly from the corresponding columns. Decision variables x_f are continuous and non-negative for all foods.