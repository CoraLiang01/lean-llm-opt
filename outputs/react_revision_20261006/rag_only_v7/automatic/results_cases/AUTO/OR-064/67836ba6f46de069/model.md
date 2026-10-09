Mathematical Model:

Sets:
- Let \( F \) be the set of suppliers, indexed by \( i \), with IDs from file_1_view_0["Unnamed: 0"] and file_2_view_0["Unnamed: 0"].
- Let \( S \) be the set of supermarkets, indexed by \( j \), with IDs from file_0_view_0["customer"] and file_2_view_0 columns (excluding "Unnamed: 0").

Parameters:
- \( f_i \): Fixed cost of opening supplier \( i \), from file_1_view_0["fixed_costs"].
- \( c_{ij} \): Transportation cost per unit from supplier \( i \) to supermarket \( j \), from file_2_view_0, row "Unnamed: 0" = \( i \), column \( j \).
- \( d_j \): Demand of supermarket \( j \), from file_0_view_0["demand"].

Decision Variables:
- \( y_i \in \{0,1\} \): 1 if supplier \( i \) is open, 0 otherwise.
- \( x_{ij} \geq 0 \): Quantity supplied from supplier \( i \) to supermarket \( j \).

Objective:
\[
\min \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in S} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction for each supermarket:
\[
\sum_{i \in F} x_{ij} = d_j \quad \forall j \in S
\]
2. Supplier can only supply if open:
\[
x_{ij} \leq d_j y_i \quad \forall i \in F, \forall j \in S
\]
3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in F
\]
\[
x_{ij} \geq 0 \quad \forall i \in F, \forall j \in S
\]

Data Mapping:
- \( F \): file_1_view_0["Unnamed: 0"], file_2_view_0["Unnamed: 0"]
- \( S \): file_0_view_0["customer"], file_2_view_0 columns (excluding "Unnamed: 0")
- \( f_i \): file_1_view_0["fixed_costs"], indexed by file_1_view_0["Unnamed: 0"]
- \( c_{ij} \): file_2_view_0, row "Unnamed: 0" = \( i \), column \( j \)
- \( d_j \): file_0_view_0["demand"], indexed by file_0_view_0["customer"]

Index sets, parameters, and all constraints are defined exactly as mapped above. No values are enumerated; all are symbolically bound to their source columns.