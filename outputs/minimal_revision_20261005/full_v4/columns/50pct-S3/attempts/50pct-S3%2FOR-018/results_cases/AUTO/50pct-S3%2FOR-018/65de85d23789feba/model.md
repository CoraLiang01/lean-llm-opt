##### Decision Variables

For each supplier $i$ (distribution center) and customer group $j$:
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer $j$ (continuous).

##### Parameters

- $I$: set of suppliers (distribution centers), from `supply_capacity.csv` (table_id: file_1_view_0), column `supplier_id`.
- $J$: set of customer groups, from `customer_demand.csv` (table_id: file_0_view_0), column `customer_id`.
- $d_j$: demand of customer $j$, from `customer_demand.csv` (table_id: file_0_view_0), column `demand`.
- $s_i$: supply capacity of supplier $i$, from `supply_capacity.csv` (table_id: file_1_view_0), column `supply_capacity`.
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$, from `transportation_costs.csv` (table_id: file_2_view_0), column `transportation_cost_to_{j}` for row with `supplier_id = i`.

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction** (each customer group receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   $$
2. **Supply capacity** (each supplier does not exceed its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   $$
3. **Non-negativity**:
   $$
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ = all `supplier_id` in `supply_capacity.csv` (table_id: file_1_view_0)
- $J$ = all `customer_id` in `customer_demand.csv` (table_id: file_0_view_0)
- $d_j$ = value in `demand` column for customer $j$ in `customer_demand.csv` (table_id: file_0_view_0)
- $s_i$ = value in `supply_capacity` column for supplier $i$ in `supply_capacity.csv` (table_id: file_1_view_0)
- $c_{ij}$ = value in `transportation_cost_to_{j}` column for row with `supplier_id = i` in `transportation_costs.csv` (table_id: file_2_view_0)

##### Index Sets

- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12$\}$
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12$\}$

##### Complete Model

Minimize
$$
\sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

subject to
$$
\sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
$$
$$
\sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
$$
$$
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
$$

with all parameters and index sets mapped exactly as above to the retrieved data.