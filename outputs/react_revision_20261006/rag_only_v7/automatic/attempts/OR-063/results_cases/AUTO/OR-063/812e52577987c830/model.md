Mathematical Model:

Sets:
- Let \( F \) be the set of warehouses, indexed by \( i \), with IDs from file_1_view_0["Unnamed: 0"].
- Let \( C \) be the set of musicians/bands (customers), indexed by \( j \), with IDs from file_0_view_0["customer"].

Parameters:
- \( f_i \): Fixed cost of opening warehouse \( i \), from file_1_view_0["fixed_costs"].
- \( d_j \): Demand of customer \( j \), from file_0_view_0["demand"].
- \( c_{ij} \): Transportation cost per unit from warehouse \( i \) to customer \( j \), from file_2_view_0, row "Unnamed: 0" = \( i \), column \( j \).

Decision Variables:
- \( y_i \in \{0,1\} \): 1 if warehouse \( i \) is open, 0 otherwise.
- \( x_{ij} \geq 0 \): Quantity supplied from warehouse \( i \) to customer \( j \).

Objective:
\[
\min \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in C} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction for each customer:
\[
\sum_{i \in F} x_{ij} = d_j \quad \forall j \in C
\]
2. Supply only from open warehouses:
\[
x_{ij} \leq d_j y_i \quad \forall i \in F,\, j \in C
\]
3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in F
\]
\[
x_{ij} \geq 0 \quad \forall i \in F,\, j \in C
\]

Data Mapping:
- \( F = \) all values in file_1_view_0["Unnamed: 0"]
- \( C = \) all values in file_0_view_0["customer"]
- \( f_i = \) file_1_view_0["fixed_costs"], indexed by file_1_view_0["Unnamed: 0"]
- \( d_j = \) file_0_view_0["demand"], indexed by file_0_view_0["customer"]
- \( c_{ij} = \) file_2_view_0, with rows indexed by "Unnamed: 0" (warehouse \( i \)), columns by customer \( j \) (column names "C1", "C2", ..., "C7")

All index sets, parameters, and constraints are mapped directly to the current CSV data as described above.