Mathematical Model

Sets:
F = set of all foods in cost.csv (indexed by f)

Parameters (from cost.csv, table_id: file_0_view_0):
cal_f = Calories per serving of food f (column: Calories)
prot_f = Protein (g) per serving of food f (column: Protein(g))
fat_f = Fat (g) per serving of food f (column: Fat(g))
vitc_f = VitaminC (mg) per serving of food f (column: VitaminC(mg))
cost_f = Cost (USD) per serving of food f (column: Cost)

Decision Variables:
x_f ≥ 0 : number of servings of food f (continuous, ∀ f ∈ F)

Objective:
Minimize total cost:
minimize   ∑_{f∈F} cost_f x_f

Subject to:
Calorie requirement:
∑_{f∈F} cal_f x_f ≥ 2000

Protein requirement:
∑_{f∈F} prot_f x_f ≥ 50

Vitamin C requirement:
∑_{f∈F} vitc_f x_f ≥ 60

Fat upper bound:
∑_{f∈F} fat_f x_f ≤ 70

Nonnegativity:
x_f ≥ 0   ∀ f ∈ F

Data Mapping:
- F: All foods in cost.csv, column Food, table_id file_0_view_0
- cal_f: Calories, file_0_view_0
- prot_f: Protein(g), file_0_view_0
- fat_f: Fat(g), file_0_view_0
- vitc_f: VitaminC(mg), file_0_view_0
- cost_f: Cost, file_0_view_0
- x_f: servings of food f (continuous, ≥ 0) for all f ∈ F

All constraints and parameters are directly mapped from the provided cost.csv file. Portions may be fractional. The model minimizes total cost while meeting all nutritional requirements.