##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from supplier $i$ to customer $j$, for all $i \in I$, $j \in J$.

##### Index Sets

- $I$ = set of suppliers = {supply1, supply2, supply3, supply4, supply5, supply6, supply7, supply8} (from `supply_capacity.csv` and `transportation_costs.csv`, column: supplier_id)
- $J$ = set of customers = {demand1, demand2, demand3, demand4, demand5, demand6, demand7, demand8} (from `customer_demand.csv` and `transportation_costs.csv`, columns: customer_id, transportation_cost_to_demand*)

##### Parameters

- $d_j$ = demand of customer $j$ (from `customer_demand.csv`, column: demand)
- $s_i$ = supply capacity of supplier $i$ (from `supply_capacity.csv`, column: supply_capacity)
- $c_{ij}$ = transportation cost per unit from supplier $i$ to customer $j$ (from `transportation_costs.csv`, columns: transportation_cost_to_demand*)

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

- $I$ (suppliers): All `supplier_id` in `supply_capacity.csv` and `transportation_costs.csv`
- $J$ (customers): All `customer_id` in `customer_demand.csv` and columns with suffix `demand*` in `transportation_costs.csv`
- $d_j$: `customer_demand.csv` (table_id: file_0_view_0), column: demand, indexed by customer_id
- $s_i$: `supply_capacity.csv` (table_id: file_1_view_0), column: supply_capacity, indexed by supplier_id
- $c_{ij}$: `transportation_costs.csv` (table_id: file_2_view_0), row: supplier_id, column: transportation_cost_to_demand*, where $i$ = supplier_id, $j$ = demand*

All indices, parameters, and coefficients are bound directly to the retrieved data as described above.