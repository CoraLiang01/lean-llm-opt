##### Decision Variables

For each supplier $i$ in $I$ and customer $j$ in $J$:
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer $j$ (continuous).

##### Objective Function

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction** (each customer receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity** (each supplier does not exceed its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity**:
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$: set of suppliers (distribution centers), from `supply_capacity.csv` and `transportation_costs.csv`, column `supplier_id`, table_id: `file_1_view_0` and `file_2_view_0`
- $J$: set of customers, from `customer_demand.csv` and `transportation_costs.csv`, column `customer_id`, table_id: `file_0_view_0` and columns with suffix in `transportation_cost_to_C*`, table_id: `file_2_view_0`
- $d_j$: demand for customer $j$, from `customer_demand.csv`, column `demand_units`, table_id: `file_0_view_0`
- $s_i$: supply capacity for supplier $i$, from `supply_capacity.csv`, column `supply_capacity_units`, table_id: `file_1_view_0`
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$, from `transportation_costs.csv`, column `transportation_cost_to_{j}`, row `supplier_id = i`, table_id: `file_2_view_0`

---

#### Index Sets

- $I = \{$S1, S2, ..., S18$\}$ (all `supplier_id` in `file_1_view_0`)
- $J = \{$C1, C2, ..., C18$\}$ (all `customer_id` in `file_0_view_0`)

---

#### Complete Model

$$
\begin{align*}
\min_{x_{ij} \geq 0} \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.} \quad
& \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\end{align*}
$$

---

#### Data Table Reference

- $I$: `supplier_id`, table_id: `file_1_view_0`
- $J$: `customer_id`, table_id: `file_0_view_0`
- $d_j$: `demand_units`, table_id: `file_0_view_0`
- $s_i$: `supply_capacity_units`, table_id: `file_1_view_0`
- $c_{ij}$: `transportation_cost_to_{j}`, row `supplier_id = i`, table_id: `file_2_view_0`