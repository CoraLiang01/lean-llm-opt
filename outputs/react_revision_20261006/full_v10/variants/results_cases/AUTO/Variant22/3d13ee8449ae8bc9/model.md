Mathematical Model

Sets:
C: set of component families (from option_catalog.csv, column Family)
O_c: set of options for family c ∈ C (from option_catalog.csv, column Option for each Family)

Parameters:
v_{c,o}: value of option o in family c (file_0_view_0, Value)
w_{c,o}: weight of option o in family c (file_0_view_0, Weight)
h_{c,o}: labor-hours of option o in family c (file_0_view_0, LaborHours)
W^{\max}: total weight limit (file_1_view_0, Limit where Resource = "Weight")
H^{\max}: total labor-hour limit (file_1_view_0, Limit where Resource = "LaborHours")

Decision Variables:
x_{c,o} ∈ {0,1} for all c ∈ C, o ∈ O_c
  x_{c,o} = 1 if option o is selected from family c, 0 otherwise

Objective:
maximize ∑_{c ∈ C} ∑_{o ∈ O_c} v_{c,o} x_{c,o}

Subject to:
1. Exactly one option per family:
  ∑_{o ∈ O_c} x_{c,o} = 1  for all c ∈ C

2. Total weight constraint:
  ∑_{c ∈ C} ∑_{o ∈ O_c} w_{c,o} x_{c,o} ≤ W^{\max}

3. Total labor-hour constraint:
  ∑_{c ∈ C} ∑_{o ∈ O_c} h_{c,o} x_{c,o} ≤ H^{\max}

4. Binary restrictions:
  x_{c,o} ∈ {0,1}  for all c ∈ C, o ∈ O_c

Data Mapping:
- C = {C1, C2, C3, C4, C5, C6}  (file_0_view_0, Family)
- O_c = {O1, O2, O3}  (file_0_view_0, Option for each Family)
- v_{c,o}, w_{c,o}, h_{c,o} from file_0_view_0, columns Value, Weight, LaborHours for each (Family, Option)
- W^{\max} = 55  (file_1_view_0, Limit where Resource = "Weight")
- H^{\max} = 64  (file_1_view_0, Limit where Resource = "LaborHours")