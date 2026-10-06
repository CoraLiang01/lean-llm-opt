##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from plant $i$ to retail outlet $j$, for all $i \in I$, $j \in J$.

##### Index Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants, from `file_1_view_0.supplier_id`)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets, from `file_0_view_0.customer_id`)

##### Parameters

- $d_j$: demand at outlet $j$ (from `file_0_view_0.demand`)
- $s_i$: supply capacity at plant $i$ (from `file_1_view_0.supply_capacity`)
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from `file_2_view_0`, columns `transportation_cost_to_C1`, ..., `transportation_cost_to_C4`)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each outlet $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
2. **Supply capacity:**  
   For each plant $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (plants): all `supplier_id` in `file_1_view_0`
- $J$ (outlets): all `customer_id` in `file_0_view_0`
- $d_j$: `file_0_view_0.demand` for $j$
- $s_i$: `file_1_view_0.supply_capacity` for $i$
- $c_{ij}$: `file_2_view_0.transportation_cost_to_{j}` for row with `supplier_id = i`

##### Table IDs and Columns

- `file_0_view_0`: columns `customer_id`, `demand`
- `file_1_view_0`: columns `supplier_id`, `supply_capacity`
- `file_2_view_0`: columns `supplier_id`, `transportation_cost_to_C1`, `transportation_cost_to_C2`, `transportation_cost_to_C3`, `transportation_cost_to_C4`