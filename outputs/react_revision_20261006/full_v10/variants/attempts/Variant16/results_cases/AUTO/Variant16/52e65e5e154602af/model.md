Mathematical Model

Sets:
G: set of families (from option_catalog.csv, column Family)
O_g: set of options for family g ∈ G (from option_catalog.csv, column Option for each Family)

Parameters:
v_{g,o}: value of option o in family g (from option_catalog.csv, column Value)
w_{g,o}: weight of option o in family g (from option_catalog.csv, column Weight)
b_{g,o}: budget use of option o in family g (from option_catalog.csv, column BudgetUse)
W^{\max}: total weight limit (from resource_limits.csv, row Resource = Weight, column Limit)
B^{\max}: total budget use limit (from resource_limits.csv, row Resource = BudgetUse, column Limit)

Decision Variables:
x_{g,o} ∈ {0,1} for all g ∈ G, o ∈ O_g
  x_{g,o} = 1 if option o is selected from family g, 0 otherwise

Objective:
maximize ∑_{g ∈ G} ∑_{o ∈ O_g} v_{g,o} x_{g,o}

Subject to:
1. Exactly one option selected per family:
  ∑_{o ∈ O_g} x_{g,o} = 1  for all g ∈ G

2. Total weight constraint:
  ∑_{g ∈ G} ∑_{o ∈ O_g} w_{g,o} x_{g,o} ≤ W^{\max}

3. Total budget use constraint:
  ∑_{g ∈ G} ∑_{o ∈ O_g} b_{g,o} x_{g,o} ≤ B^{\max}

4. Binary restrictions:
  x_{g,o} ∈ {0,1}  for all g ∈ G, o ∈ O_g

Data Mapping:
- G and O_g: All unique Family and Option pairs in option_catalog.csv (table_id: file_0_view_0, columns: Family, Option)
- v_{g,o}: option_catalog.csv (table_id: file_0_view_0, column: Value)
- w_{g,o}: option_catalog.csv (table_id: file_0_view_0, column: Weight)
- b_{g,o}: option_catalog.csv (table_id: file_0_view_0, column: BudgetUse)
- W^{\max}: resource_limits.csv (table_id: file_1_view_0, row Resource = Weight, column Limit)
- B^{\max}: resource_limits.csv (table_id: file_1_view_0, row Resource = BudgetUse, column Limit)