##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the data.

Decision variables:
$$
x_{ij} \geq 0 \quad \text{(continuous)}, \quad \forall i \in I,\, j \in J
$$
where $x_{ij}$ is the quantity shipped from distribution center $i$ to customer group $j$.

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$
where $c_{ij}$ is the transportation cost per unit from supplier $i$ to customer $j$.

Subject to:
1. Demand satisfaction:
$$
\sum_{i \in I} x_{ij} \geq d_j, \quad \forall j \in J
$$
where $d_j$ is the demand of customer group $j$.

2. Supply capacity:
$$
\sum_{j \in J} x_{ij} \leq s_i, \quad \forall i \in I
$$
where $s_i$ is the supply capacity of supplier $i$.

3. Non-negativity:
$$
x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
$$

##### Data Mapping

- $I$ (distribution centers/suppliers): All unique `supplier_id` in `supply_capacity.csv` and `transportation_costs.csv` (S1, S2, ..., S12).
- $J$ (customer groups): All unique `customer_id` in `customer_demand.csv` and columns in `transportation_costs.csv` (C1, C2, ..., C12).
- $d_j$: For each $j \in J$, $d_j$ is the value in column `demand` of `customer_demand.csv` where `customer_id` = $j$ (table_id: file_0_view_0).
- $s_i$: For each $i \in I$, $s_i$ is the value in column `supply_capacity` of `supply_capacity.csv` where `supplier_id` = $i$ (table_id: file_1_view_0).
- $c_{ij}$: For each $i \in I$, $j \in J$, $c_{ij}$ is the value in `transportation_costs.csv` at row with `supplier_id` = $i$ and column `transportation_cost_to_{j}` (table_id: file_2_view_0).

Index sets, parameters, and all coefficients are defined exactly by the current CSV data. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.