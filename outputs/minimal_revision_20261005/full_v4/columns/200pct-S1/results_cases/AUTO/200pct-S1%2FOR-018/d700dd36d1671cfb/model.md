##### Decision Variables

For each supplier $i$ (distribution center) and customer group $j$:
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer $j$ (continuous).

##### Parameters

- $I$: set of suppliers (distribution centers), from `supply_capacity.csv`, column `supplier_id`, table_id: `file_1_view_0`.
- $J$: set of customer groups, from `customer_demand.csv`, column `customer_id`, table_id: `file_0_view_0`.
- $d_j$: demand of customer $j$, from `customer_demand.csv`, column `demand`, table_id: `file_0_view_0`.
- $s_i$: supply capacity of supplier $i$, from `supply_capacity.csv`, column `supply_capacity`, table_id: `file_1_view_0`.
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$, from `transportation_costs.csv`, column `transportation_cost_to_{j}`, table_id: `file_2_view_0`.

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

---

#### Data Mapping

- $I$ (suppliers): all `supplier_id` in `file_1_view_0` (`supply_capacity.csv`)
- $J$ (customers): all `customer_id` in `file_0_view_0` (`customer_demand.csv`)
- $d_j$: `demand` for customer $j$ in `file_0_view_0` (`customer_demand.csv`)
- $s_i$: `supply_capacity` for supplier $i$ in `file_1_view_0` (`supply_capacity.csv`)
- $c_{ij}$: `transportation_cost_to_{j}` for supplier $i$ in `file_2_view_0` (`transportation_costs.csv`), where `{j}` matches `customer_id` in `file_0_view_0`

All indices, parameters, and coefficients are to be taken exactly as listed in the respective files and columns, using the full set of returned entities.