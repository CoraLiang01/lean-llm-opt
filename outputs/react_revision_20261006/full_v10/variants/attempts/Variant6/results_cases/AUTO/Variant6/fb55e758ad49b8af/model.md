Mathematical Model

Sets:
F = set of product families (Family in file_0_view_0)
O_f = set of options for family f (Option in file_0_view_0 where Family = f)
R = {Weight, BudgetUse} (Resource in file_1_view_0)

Parameters:
v_{fo} = Value of option o in family f (Value, file_0_view_0)
w_{fo} = Weight of option o in family f (Weight, file_0_view_0)
b_{fo} = BudgetUse of option o in family f (BudgetUse, file_0_view_0)
L_r = Limit for resource r (Limit, file_1_view_0)

Decision variables:
x_{fo} ∈ {0,1} for all f ∈ F, o ∈ O_f

Objective:
maximize ∑_{f∈F} ∑_{o∈O_f} v_{fo} x_{fo}

Subject to:
1. Exactly one option per family:
  ∑_{o∈O_f} x_{fo} = 1  for all f ∈ F

2. Resource limits:
  ∑_{f∈F} ∑_{o∈O_f} w_{fo} x_{fo} ≤ L_Weight
  ∑_{f∈F} ∑_{o∈O_f} b_{fo} x_{fo} ≤ L_BudgetUse

3. Binary restrictions:
  x_{fo} ∈ {0,1} for all f ∈ F, o ∈ O_f

Data Mapping

Sets:
F = unique values of Family in file_0_view_0
O_f = unique values of Option in file_0_view_0 where Family = f
R = Resource in file_1_view_0

Parameters:
v_{fo} = Value (file_0_view_0, columns Family, Option, Value)
w_{fo} = Weight (file_0_view_0, columns Family, Option, Weight)
b_{fo} = BudgetUse (file_0_view_0, columns Family, Option, BudgetUse)
L_r = Limit (file_1_view_0, columns Resource, Limit)

Variables:
x_{fo} for each (Family, Option) pair in file_0_view_0

Constraints:
- One per family: sum over Option where Family = f
- Resource limits: sum over all (Family, Option) pairs, using Weight and BudgetUse columns, bounded by corresponding Limit in file_1_view_0

Objective:
- Maximize sum over all (Family, Option) pairs of Value × x_{fo}