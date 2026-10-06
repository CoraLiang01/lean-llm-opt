##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from supplier $i$ to customer $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I$: set of suppliers (distribution centers), from `supply_capacity.csv` and `transportation_costs.csv` rows:
  $I = \{\text{supply1}, \text{supply2}, \text{supply3}, \text{supply4}, \text{supply5}, \text{supply6}, \text{supply7}, \text{supply8}\}$
- $J$: set of customers (customer groups), from `customer_demand.csv` and `transportation_costs.csv` columns:
  $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

##### Parameters

- $d_j$: demand of customer $j$, from `customer_demand.csv` (`file_0_view_0`), column `demand`
- $s_i$: supply capacity of supplier $i$, from `supply_capacity.csv` (`file_1_view_0`), column `supply_capacity`
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$, from `transportation_costs.csv` (`file_2_view_0`), column `transportation_cost_to_{j}`

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   $$
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   $$
2. **Supply capacity:**  
   $$
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (suppliers): all `supplier_id` in `supply_capacity.csv` (`file_1_view_0`) and `transportation_costs.csv` (`file_2_view_0`)
- $J$ (customers): all `customer_id` in `customer_demand.csv` (`file_0_view_0`) and columns `transportation_cost_to_{customer_id}` in `transportation_costs.csv` (`file_2_view_0`)
- $d_j$: `customer_demand.csv` (`file_0_view_0`), column `demand`, row where `customer_id = j`
- $s_i$: `supply_capacity.csv` (`file_1_view_0`), column `supply_capacity`, row where `supplier_id = i`
- $c_{ij}$: `transportation_costs.csv` (`file_2_view_0`), value in row where `supplier_id = i`, column `transportation_cost_to_{j}`

All indices, parameters, and coefficients are bound exactly to the retrieved data and identifiers.