##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i$ to customer $j$.

- $i \in I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $j \in J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

##### Objective Function

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
   where $d_j$ is the demand for customer $j$.

2. **Supply capacity:**  
   For each supplier $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
   where $s_i$ is the supply capacity of supplier $i$.

3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- **Suppliers ($I$):**  
  supplier1, supplier2, supplier3, supplier4, supplier5, supplier6, supplier7, supplier8  
  (from supply_capacity.csv, source order)

- **Customers ($J$):**  
  demand1, demand2, demand3, demand4, demand5, demand6, demand7, demand8  
  (from customer_demand.csv, source order)

- **Demand ($d_j$):**  
  demand1: 9  
  demand2: 66  
  demand3: 56  
  demand4: 17  
  demand5: 43  
  demand6: 62  
  demand7: 10  
  demand8: 37  
  (from customer_demand.csv, source order)

- **Supply capacity ($s_i$):**  
  supplier1: 60  
  supplier2: 22  
  supplier3: 16  
  supplier4: 14  
  supplier5: 19  
  supplier6: 70  
  supplier7: 60  
  supplier8: 39  
  (from supply_capacity.csv, source order)

- **Transportation cost ($c_{ij}$):**  
  $c_{ij}$ is the entry in transportation_costs.csv at row $i$ (supplier, source order) and column $j$ (customer, source order).  
  For example, $c_{\text{supplier1},\text{demand1}} = 0.03020736643461065$, $c_{\text{supplier2},\text{demand3}} = 0.28605434149404785$, etc.  
  (from transportation_costs.csv, source order for both suppliers and customers)

---

**All indices, coefficients, and constraints are mapped directly from the retrieved data in source order.**