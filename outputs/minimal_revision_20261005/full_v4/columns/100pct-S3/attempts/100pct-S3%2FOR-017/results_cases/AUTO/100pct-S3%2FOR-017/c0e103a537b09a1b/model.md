##### Decision Variables

For each supplier $i$ in $I$ and customer $j$ in $J$:
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer $j$ (continuous).

##### Sets

- $I$: set of suppliers, from `supply_capacity.csv` (table_id: file_1_view_0), column `supplier_id`:
  $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$
- $J$: set of customers, from `customer_demand.csv` (table_id: file_0_view_0), column `customer_id`:
  $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

##### Parameters

- $d_j$: demand of customer $j$, from `customer_demand.csv` (table_id: file_0_view_0), column `demand`, indexed by `customer_id`.
- $s_i$: supply capacity of supplier $i$, from `supply_capacity.csv` (table_id: file_1_view_0), column `supply_capacity`, indexed by `supplier_id`.
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$, from `transportation_costs.csv` (table_id: file_2_view_0), column `transportation_cost_to_{j}`, row `supplier_id`.

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
2. **Supply capacity** (each supplier ships no more than its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   $$
3. **Non-negativity**:
   $$
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (suppliers): all `supplier_id` in `supply_capacity.csv` (table_id: file_1_view_0)
- $J$ (customers): all `customer_id` in `customer_demand.csv` (table_id: file_0_view_0)
- $d_j$: `demand` from `customer_demand.csv` (table_id: file_0_view_0), indexed by `customer_id`
- $s_i$: `supply_capacity` from `supply_capacity.csv` (table_id: file_1_view_0), indexed by `supplier_id`
- $c_{ij}$: `transportation_cost_to_{j}` from `transportation_costs.csv` (table_id: file_2_view_0), row `supplier_id`, column for each customer $j$ (see relationships in Observation)

All indices, parameters, and coefficients are bound directly to the retrieved data as specified above.