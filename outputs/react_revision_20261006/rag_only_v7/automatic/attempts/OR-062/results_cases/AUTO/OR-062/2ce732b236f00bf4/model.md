Mathematical Model:

Sets:
- Let F be the set of suppliers, indexed by i. (F = {MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES}, from file_1_view_0 and file_2_view_0 rows)
- Let S be the set of stores (customers), indexed by j. (S = {CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT}, from file_2_view_0 columns and file_0_view_0 rows)

Parameters:
- d_j: demand of store j. (from file_0_view_0, column "demand", for Customer = j)
- f_i: fixed cost to open supplier i. (from file_1_view_0, column "fixed_costs", for Unnamed: 0 = i)
- c_{ij}: transportation cost per unit from supplier i to store j. (from file_2_view_0, row Unnamed: 0 = i, column j)

Decision Variables:
- y_i ∈ {0,1}: 1 if supplier i is open, 0 otherwise.
- x_{ij} ≥ 0: quantity supplied from supplier i to store j.

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
2. Non-negativity:
\[
x_{ij} \geq 0 \quad \forall i \in F, j \in S
\]
3. Supplier activation (no explicit capacity, so no further constraint):
\[
y_i \in \{0,1\} \quad \forall i \in F
\]

Data Mapping:
- F (suppliers): file_1_view_0["Unnamed: 0"] and file_2_view_0["Unnamed: 0"]
- S (stores): file_2_view_0 columns ["CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"] and file_0_view_0["Customer"]
- d_j: file_0_view_0, column "demand", for Customer = j
- f_i: file_1_view_0, column "fixed_costs", for Unnamed: 0 = i
- c_{ij}: file_2_view_0, row Unnamed: 0 = i, column j

Variables:
- y_i: binary, for each i ∈ F
- x_{ij}: continuous, for each i ∈ F, j ∈ S

Constraints:
- For each j ∈ S: sum_{i ∈ F} x_{ij} = d_j
- For each i ∈ F, j ∈ S: x_{ij} ≥ 0
- For each i ∈ F: y_i ∈ {0,1}

Objective:
- Minimize sum_{i ∈ F} f_i y_i + sum_{i ∈ F} sum_{j ∈ S} c_{ij} x_{ij}

All parameter values are mapped directly from the specified columns and rows in the current CSV files.