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

- $I$ (suppliers):  
  From `supply_capacity.csv` and `transportation_costs.csv`, column: `supplier_id`  
  Values: supply1, supply2, supply3, supply4, supply5, supply6, supply7, supply8

- $J$ (customers):  
  From `customer_demand.csv`, column: `customer_id`  
  Values: demand1, demand2, demand3, demand4, demand5, demand6, demand7, demand8

- $d_j$:  
  From `customer_demand.csv`, column: `demand`  
  Mapping:  
  demand1: 9  
  demand2: 66  
  demand3: 56  
  demand4: 17  
  demand5: 43  
  demand6: 62  
  demand7: 10  
  demand8: 37  

- $s_i$:  
  From `supply_capacity.csv`, column: `supply_capacity`  
  Mapping:  
  supply1: 60  
  supply2: 22  
  supply3: 16  
  supply4: 14  
  supply5: 19  
  supply6: 70  
  supply7: 60  
  supply8: 39  

- $c_{ij}$:  
  From `transportation_costs.csv`, columns: `transportation_cost_to_demand1`, ..., `transportation_cost_to_demand8`  
  For each $i$ (row: supplier_id), $j$ (column: transportation_cost_to_demand*)  
  Example: $c_{\text{supply1},\text{demand1}} = 0.03020736643461065$, $c_{\text{supply2},\text{demand3}} = 0.28605434149404785$, etc.

---

**All indices, parameters, and coefficients are bound exactly to the retrieved data and identifiers.**