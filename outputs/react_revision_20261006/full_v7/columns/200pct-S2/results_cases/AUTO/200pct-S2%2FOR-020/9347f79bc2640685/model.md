##### Mathematical Model

Let $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (warehouses), $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$ (stores).

Decision variables:
$$
x_{ij} \geq 0 \quad \text{(continuous)}, \quad \forall i \in I,\, j \in J
$$
where $x_{ij}$ is the quantity shipped from warehouse $i$ to store $j$.

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
1. Store demand satisfaction:
$$
\sum_{i \in I} x_{ij} \geq d_j, \quad \forall j \in J
$$

2. Warehouse supply capacity:
$$
\sum_{j \in J} x_{ij} \leq s_i, \quad \forall i \in I
$$

3. Non-negativity:
$$
x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
$$

##### Data Mapping

- $I$ (warehouses): supplier_id from supply_capacity.csv and transportation_costs.csv: S1, S2, S3, S4, S5
- $J$ (stores): customer_id from customer_demand.csv and transportation_costs.csv: D1, D2, D3, D4, D5
- $d_j$: demand_units for store $j$ from customer_demand.csv (table_id: file_0_view_0, columns: customer_id, demand_units)
- $s_i$: supply_capacity_units for warehouse $i$ from supply_capacity.csv (table_id: file_1_view_0, columns: supplier_id, supply_capacity_units)
- $c_{ij}$: transportation_cost_to_Dk for warehouse $i$ to store $j$ from transportation_costs.csv (table_id: file_2_view_0, columns: supplier_id, transportation_cost_to_D1, ..., transportation_cost_to_D5; row: supplier_id $i$; column: Dk for $j$)

Index sets, parameters, and all coefficients are defined by the current CSV data as described above.