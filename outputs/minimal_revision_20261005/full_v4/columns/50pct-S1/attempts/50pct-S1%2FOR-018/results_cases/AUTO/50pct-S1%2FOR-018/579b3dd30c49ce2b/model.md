##### Decision Variables

For each supplier $i$ (distribution center) and customer group $j$:
$$
x_{ij} \geq 0
$$
where $x_{ij}$ is the continuous quantity shipped from supplier $i$ to customer $j$.

##### Parameters

- $I$: set of suppliers (distribution centers), from column `supplier_id` in `supply_capacity.csv` and `transportation_costs.csv`.
- $J$: set of customers, from column `customer_id` in `customer_demand.csv` and columns `transportation_cost_to_{Ck}` in `transportation_costs.csv`.
- $d_j$: demand of customer $j$, from `demand` in `customer_demand.csv`.
- $s_i$: supply capacity of supplier $i$, from `supply_capacity` in `supply_capacity.csv`.
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$, from `transportation_cost_to_{j}` in `transportation_costs.csv`.

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction** (each customer receives at least its demand):
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

---

#### Data Mapping

- $I$ (suppliers): all `supplier_id` in `supply_capacity.csv` (`file_1_view_0`) and `transportation_costs.csv` (`file_2_view_0`)
- $J$ (customers): all `customer_id` in `customer_demand.csv` (`file_0_view_0`) and columns `transportation_cost_to_{Ck}` in `transportation_costs.csv` (`file_2_view_0`)
- $d_j$: `demand` for customer $j$ from `customer_demand.csv` (`file_0_view_0`)
- $s_i$: `supply_capacity` for supplier $i$ from `supply_capacity.csv` (`file_1_view_0`)
- $c_{ij}$: value in `transportation_costs.csv` (`file_2_view_0`), row `supplier_id` $i$, column `transportation_cost_to_{j}`

##### Index Sets

- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12$\}$
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12$\}$

##### Complete Model

$$
\begin{align*}
\min_{x_{ij} \geq 0} \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.} \quad
& \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I \\
& x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
\end{align*}
$$

##### Data Table Mapping

- $d_j$: `file_0_view_0`, columns: `customer_id`, `demand`
- $s_i$: `file_1_view_0`, columns: `supplier_id`, `supply_capacity`
- $c_{ij}$: `file_2_view_0`, row: `supplier_id`, columns: `transportation_cost_to_{Ck}`

All index sets, parameters, and constraints are bound exactly to the retrieved data.