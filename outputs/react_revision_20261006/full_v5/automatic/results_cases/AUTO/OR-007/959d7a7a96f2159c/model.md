##### Mathematical Model

Let $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (warehouses), $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$ (stores).

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:

1. **Demand satisfaction** (each store receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity** (each warehouse ships no more than its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity**:
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (warehouses): region column in supply_capacity.csv and Unnamed: 0 in transportation_costs.csv: S1, S2, S3, S4, S5
- $J$ (stores): customer column in customer_demand.csv and D1–D5 columns in transportation_costs.csv: D1, D2, D3, D4, D5
- $d_j$: demand for store $j$ from customer_demand.csv, column demand, table_id file_0_view_0
- $s_i$: supply_capacity for warehouse $i$ from supply_capacity.csv, column supply_capacity, table_id file_1_view_0
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ from transportation_costs.csv, table_id file_2_view_0, row Unnamed: 0 = $i$, column $j$

All indices, parameters, and coefficients are mapped directly from the retrieved CSV data.