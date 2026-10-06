Mathematical Model

Sets:
- Let F be the set of suppliers, indexed by i, where F = {MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES} (from file_1_view_0 and file_2_view_0, column "Unnamed: 0").
- Let S be the set of stores, indexed by j, where S = {CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT} (from file_2_view_0, columns "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT").

Parameters:
- Fixed cost for supplier i: f_i = [file_1_view_0, column "fixed_costs", row i]
- Demand at store j: d_j = [file_0_view_0, column "demand", row j]
- Transportation cost per unit from supplier i to store j: c_{ij} = [file_2_view_0, row i, column j]

Decision Variables:
- y_i ∈ {0,1} for all i ∈ F (indicates if supplier i is open)
- x_{ij} ≥ 0 for all i ∈ F, j ∈ S (quantity supplied from i to j)

Objective:
Minimize total cost:
\[
\min \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in S} c_{ij} x_{ij}
\]

Subject to:

1. Demand satisfaction at each store:
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

Data Mapping

- F (suppliers): file_1_view_0, column "Unnamed: 0" and file_2_view_0, column "Unnamed: 0"
- S (stores): file_2_view_0, columns "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"
- f_i: file_1_view_0, column "fixed_costs", row i
- d_j: file_0_view_0, column "demand", row j (mapping of store names to demand must be established externally)
- c_{ij}: file_2_view_0, row i (supplier), column j (store)

All indices, parameters, and constraints are bound directly to the supplied data tables as specified above.