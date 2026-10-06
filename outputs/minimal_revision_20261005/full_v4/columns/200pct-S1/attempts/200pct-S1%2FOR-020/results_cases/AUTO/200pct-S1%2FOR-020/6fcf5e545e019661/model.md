##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$.

##### Index Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (warehouses, from `file_1_view_0.supplier_id`)
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$ (stores, from `file_0_view_0.customer_id`)

##### Parameters

- Demand for each store $j$: $d_j$ from `file_0_view_0.demand_units`
- Supply capacity for each warehouse $i$: $s_i$ from `file_1_view_0.supply_capacity_units`
- Transportation cost per unit from warehouse $i$ to store $j$: $c_{ij}$ from `file_2_view_0`, column `"transportation_cost_to_" + j`, row `supplier_id = i`

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
   where $d_j$ is from `file_0_view_0.demand_units` for $customer_id = j$.

2. **Supply capacity:**  
   For each warehouse $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
   where $s_i$ is from `file_1_view_0.supply_capacity_units` for $supplier_id = i$.

3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (warehouses): all `supplier_id` in `file_1_view_0`
- $J$ (stores): all `customer_id` in `file_0_view_0`
- $d_j$: `file_0_view_0.demand_units` for $j$
- $s_i$: `file_1_view_0.supply_capacity_units` for $i$
- $c_{ij}$: `file_2_view_0` where `supplier_id = i`, column `"transportation_cost_to_" + j`

---

**All indices, parameters, and coefficients are bound exactly to the retrieved data and identifiers.**