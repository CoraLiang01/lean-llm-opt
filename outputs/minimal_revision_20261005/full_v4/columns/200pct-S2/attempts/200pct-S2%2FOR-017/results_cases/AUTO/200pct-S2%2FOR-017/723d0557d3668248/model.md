##### Decision Variables

For each supplier $i$ in $I$ and customer group $j$ in $J$:
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer group $j$ (continuous).

##### Objective Function

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction** (each customer group receives at least its demand):
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

- $I$ (Suppliers): All `supplier_id` in `supply_capacity.csv` (`file_1_view_0`)
- $J$ (Customer groups): All `customer_id` in `customer_demand.csv` (`file_0_view_0`)
- $d_j$: Demand for customer $j$ from `demand` in `customer_demand.csv` (`file_0_view_0`)
- $s_i$: Supply capacity for supplier $i$ from `supply_capacity` in `supply_capacity.csv` (`file_1_view_0`)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to customer $j$ from `transportation_cost_to_Ck` in `transportation_costs.csv` (`file_2_view_0`), where $k$ matches $j$.

##### Index Sets

- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10$\}$
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10$\}$

##### Parameter Bindings

- $d_j$ = value in column `demand` for row with `customer_id` = $j$ in `file_0_view_0`
- $s_i$ = value in column `supply_capacity` for row with `supplier_id` = $i$ in `file_1_view_0`
- $c_{ij}$ = value in column `transportation_cost_to_{j}` for row with `supplier_id` = $i$ in `file_2_view_0`

---

**All data, indices, and coefficients are bound exactly as retrieved from the source files and columns above.**