##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i$ to customer group $j$.

- $i \in I =$ all supplier IDs from `supply_capacity.csv` and `transportation_costs.csv`:
  $I = \{\text{supply1}, \text{supply2}, \text{supply3}, \text{supply4}, \text{supply5}, \text{supply6}, \text{supply7}, \text{supply8}\}$
- $j \in J =$ all customer IDs from `customer_demand.csv` and `transportation_costs.csv`:
  $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

##### Parameters

- $d_j$: demand for customer $j$ (from `customer_demand.csv`, column `demand`)
- $s_i$: supply capacity of supplier $i$ (from `supply_capacity.csv`, column `supply_capacity`)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from `transportation_costs.csv`, column `transportation_cost_to_{j}` for row $i$)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
2. **Supply capacity:**  
   For each supplier $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (suppliers): all `supplier_id` in `supply_capacity.csv` (`file_1_view_0`, column `supplier_id`) and `transportation_costs.csv` (`file_2_view_0`, column `supplier_id`)
- $J$ (customers): all `customer_id` in `customer_demand.csv` (`file_0_view_0`, column `customer_id`) and all columns with prefix `transportation_cost_to_` in `transportation_costs.csv` (`file_2_view_0`)
- $d_j$: `customer_demand.csv` (`file_0_view_0`), column `demand`, indexed by `customer_id`
- $s_i$: `supply_capacity.csv` (`file_1_view_0`), column `supply_capacity`, indexed by `supplier_id`
- $c_{ij}$: `transportation_costs.csv` (`file_2_view_0`), value in column `transportation_cost_to_{j}` for row with `supplier_id = i`

All indices, parameters, and coefficients are bound exactly to the retrieved data and identifiers.