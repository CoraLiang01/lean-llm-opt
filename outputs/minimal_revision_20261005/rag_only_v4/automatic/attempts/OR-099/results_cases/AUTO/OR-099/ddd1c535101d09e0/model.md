Mathematical Model

Sets:
- Let 𝑊 be the set of warehouses, indexed by i. (𝑊 = all "Warehouse (i)" in table_id: file_0_view_0)
- Let 𝑆 be the set of stores, indexed by j. (𝑆 = all "Store (j)" in table_id: file_1_view_0)

Parameters:
- 𝑓ᵢ: Opening cost of warehouse i. (from "Opening Cost (fi)", table_id: file_0_view_0, for each i ∈ 𝑊)
- 𝐶ᵢ: Capacity of warehouse i. (from "Capacity (units)", table_id: file_0_view_0, for each i ∈ 𝑊)
- 𝑑ⱼ: Demand of store j. (from "Demand (units, dj)", table_id: file_1_view_0, for each j ∈ 𝑆)
- 𝑐ᵢⱼ: Transportation cost per unit from warehouse i to store j. (from table_id: file_2_view_0, row "Warehouse (i)" = i, column "Wk" where k = j, for all i ∈ 𝑊, j ∈ 𝑆)

Decision Variables:
- 𝑦ᵢ ∈ {0,1}: 1 if warehouse i is opened, 0 otherwise, ∀i ∈ 𝑊
- 𝑥ᵢⱼ ≥ 0: Amount shipped from warehouse i to store j, ∀i ∈ 𝑊, j ∈ 𝑆

Objective:
Minimize total cost:
\[
\min \sum_{i \in W} f_i y_i + \sum_{i \in W} \sum_{j \in S} c_{ij} x_{ij}
\]

Subject to:

1. Demand satisfaction for each store:
\[
\sum_{i \in W} x_{ij} = d_j \quad \forall j \in S
\]

2. Warehouse capacity (only if opened):
\[
\sum_{j \in S} x_{ij} \leq C_i y_i \quad \forall i \in W
\]

3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in W
\]
\[
x_{ij} \geq 0 \quad \forall i \in W, j \in S
\]

Data Mapping

- 𝑊: All "Warehouse (i)" in table_id: file_0_view_0
- 𝑆: All "Store (j)" in table_id: file_1_view_0
- 𝑓ᵢ: "Opening Cost (fi)" in table_id: file_0_view_0, for i
- 𝐶ᵢ: "Capacity (units)" in table_id: file_0_view_0, for i
- 𝑑ⱼ: "Demand (units, dj)" in table_id: file_1_view_0, for j
- 𝑐ᵢⱼ: Entry in table_id: file_2_view_0, row "Warehouse (i)" = i, column "Wk" where k = j

Index mapping for 𝑐ᵢⱼ:
- For warehouse i (from "Warehouse (i)" in file_0_view_0), and store j (from "Store (j)" in file_1_view_0), use row "Warehouse (i)" = i and column "Wj" in table_id: file_2_view_0.

All sets, parameters, and constraints are defined exactly as per the provided data.