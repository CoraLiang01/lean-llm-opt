##### Decision Variables

For each supplier $i$ in $I$ and customer $j$ in $J$:
$$
x_{ij} \geq 0
$$
where $x_{ij}$ is the continuous quantity shipped from supplier $i$ to customer $j$.

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

- $I$ (Suppliers): All `supplier_id` in `supply_capacity.csv` (`file_1_view_0`)
- $J$ (Customers): All `customer_id` in `customer_demand.csv` (`file_0_view_0`)
- $d_j$: Demand for customer $j$ from `demand_units` in `customer_demand.csv` (`file_0_view_0`)
- $s_i$: Supply capacity for supplier $i$ from `supply_capacity_units` in `supply_capacity.csv` (`file_1_view_0`)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to customer $j$ from `transportation_cost_to_Ck` columns in `transportation_costs.csv` (`file_2_view_0`), where $k$ matches $j$.

##### Table and Column Reference

- $I$: `supplier_id` in table_id `file_1_view_0`
- $J$: `customer_id` in table_id `file_0_view_0`
- $d_j$: `demand_units` in table_id `file_0_view_0`
- $s_i$: `supply_capacity_units` in table_id `file_1_view_0`
- $c_{ij}$: `transportation_cost_to_Ck` in table_id `file_2_view_0`, with row `supplier_id` $i$ and column $j$ as $Ck$

---

**All index sets, parameters, and coefficients are bound exactly to the retrieved data and identifiers.**