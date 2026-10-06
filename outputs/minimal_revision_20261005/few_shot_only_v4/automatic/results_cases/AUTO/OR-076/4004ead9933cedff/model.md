##### Decision Variables

- $x_{ij} \geq 0$: Amount shipped from warehouse $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise (binary).

##### Parameters

- $I = \{$W1, W2, W3, W4, W5, W6, W7, W8, W9, W10$\}$: Set of warehouses.
- $J = \{$C1, C2, ..., C20$\}$: Set of customers.
- $f_i$: Fixed cost of opening warehouse $i$.
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to customer $j$.
- $d_j$: Demand of customer $j$.
- $u_i$: Capacity of warehouse $i$.

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   $\sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J$

2. **Warehouse capacity:**  
   $\sum_{j \in J} x_{ij} \leq u_i y_i, \quad \forall i \in I$

3. **Variable domains:**  
   $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

---

#### Data Mapping

- **Warehouses ($I$):**  
  W1, W2, W3, W4, W5, W6, W7, W8, W9, W10  
  (from `warehouse.csv`, column "Warehouse ID")

- **Customers ($J$):**  
  C1, C2, ..., C20  
  (from `cost.csv` and `demand.csv`, columns "C1"..."C20" and "Customer ID")

- **Fixed costs ($f_i$):**  
  (from `warehouse.csv`, column "Fixed_Cost")  
  - W1: 2000  
  - W2: 2500  
  - W3: 1800  
  - W4: 3200  
  - W5: 1500  
  - W6: 4000  
  - W7: 2800  
  - W8: 1950  
  - W9: 3500  
  - W10: 2200  

- **Capacities ($u_i$):**  
  (from `warehouse.csv`, column "Capacity")  
  - W1: 1000  
  - W2: 1500  
  - W3: 1200  
  - W4: 2000  
  - W5: 800  
  - W6: 2500  
  - W7: 1800  
  - W8: 1100  
  - W9: 2100  
  - W10: 1300  

- **Demands ($d_j$):**  
  (from `demand.csv`, column "Demand")  
  - C1: 800  
  - C2: 600  
  - C3: 500  
  - C4: 700  
  - C5: 450  
  - C6: 950  
  - C7: 350  
  - C8: 850  
  - C9: 400  
  - C10: 750  
  - C11: 900  
  - C12: 550  
  - C13: 650  
  - C14: 820  
  - C15: 480  
  - C16: 920  
  - C17: 320  
  - C18: 780  
  - C19: 520  
  - C20: 680  

- **Transportation costs ($c_{ij}$):**  
  (from `cost.csv`, rows indexed by "Warehouse ID", columns "C1"..."C20")

---

#### Source-Column Data Mapping

- **Warehouse IDs, Fixed_Cost, Capacity:** `/warehouse.csv` columns "Warehouse ID", "Fixed_Cost", "Capacity"
- **Customer IDs, Demand:** `/demand.csv` columns "Customer ID", "Demand"
- **Transportation Costs:** `/cost.csv` rows "Warehouse ID", columns "C1"..."C20"