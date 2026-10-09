Mathematical Model:

Sets:
- Let \( F \) be the set of suppliers, indexed by \( i \), with IDs from column "Unnamed: 0" in table_id "file_1_view_0" and "file_2_view_0".
- Let \( S \) be the set of supermarkets, indexed by \( j \), with IDs from column "customer" in table_id "file_0_view_0" and columns in "file_2_view_0" (excluding "Unnamed: 0").

Parameters:
- \( f_i \): Fixed cost of opening supplier \( i \), from "fixed_costs" in table_id "file_1_view_0".
- \( c_{ij} \): Transportation cost per unit from supplier \( i \) to supermarket \( j \), from table_id "file_2_view_0", row "Unnamed: 0" = \( i \), column \( j \).
- \( d_j \): Demand of supermarket \( j \), from "demand" in table_id "file_0_view_0".

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
2. Supply only from open suppliers:
\[
x_{ij} \leq d_j y_i \quad \forall i \in F,\, j \in S
\]
3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in F
\]
\[
x_{ij} \geq 0 \quad \forall i \in F,\, j \in S
\]

Data Mapping:
- \( F \): All values in "Unnamed: 0" column of table_id "file_1_view_0" (supplier IDs S1, S2, ..., S12).
- \( S \): All values in "customer" column of table_id "file_0_view_0" (supermarket IDs C1, C2, ..., C12).
- \( f_i \): "fixed_costs" column in table_id "file_1_view_0", indexed by "Unnamed: 0".
- \( c_{ij} \): Table_id "file_2_view_0", row "Unnamed: 0" = \( i \), column \( j \).
- \( d_j \): "demand" column in table_id "file_0_view_0", indexed by "customer".

All indices, parameters, and constraints are mapped directly to the current CSV data as described above.