Mathematical Model (Uncapacitated Facility Location Problem):

Sets:
- Let I be the set of suppliers, indexed by i, with I = {S1, S2} (from file_1_view_0, column "Unnamed: 0").
- Let J be the set of supermarkets, indexed by j, with J = {C1, C2} (from file_0_view_0, column "customer").

Parameters:
- f_i: Fixed cost to activate supplier i. Data Mapping: file_1_view_0, columns "Unnamed: 0" (i), "fixed_costs".
- d_j: Demand of supermarket j. Data Mapping: file_0_view_0, columns "customer" (j), "demand".
- c_{ij}: Per-unit transportation cost from supplier i to supermarket j. Data Mapping: file_2_view_0, rows "Unnamed: 0" (i), columns "C1", "C2" (j).

Decision Variables:
- y_i ∈ {0,1}: 1 if supplier i is activated, 0 otherwise, ∀i∈I.
- x_{ij} ≥ 0: Amount supplied from supplier i to supermarket j, ∀i∈I, j∈J.

Objective:
Minimize total cost:
\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction at each supermarket:
\[
\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
\]
2. Supply only from activated suppliers:
\[
x_{ij} \leq d_j y_i \quad \forall i \in I, j \in J
\]
3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in I
\]
\[
x_{ij} \geq 0 \quad \forall i \in I, j \in J
\]

Data Mapping:
- I = {S1, S2} from file_1_view_0, column "Unnamed: 0"
- J = {C1, C2} from file_0_view_0, column "customer"
- f_i: file_1_view_0, columns "Unnamed: 0" (i), "fixed_costs"
- d_j: file_0_view_0, columns "customer" (j), "demand"
- c_{ij}: file_2_view_0, rows "Unnamed: 0" (i), columns "C1", "C2" (j)

All parameters and index sets are bound exactly to the current CSV data.