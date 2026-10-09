##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from supplier $i$ to customer $j$, for all suppliers $i \in I$ and customers $j \in J$.

##### Sets

- $I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

##### Parameters

- $d_j$: demand of customer $j$ (from customer_demand.csv)
- $s_i$: supply capacity of supplier $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from transportation_costs.csv)

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

- **customer_demand.csv**  
  - $J$ (customer indices): demand1, demand2, demand3, demand4, demand5, demand6, demand7, demand8  
  - $d_j$ values:  
    - demand1: 9  
    - demand2: 66  
    - demand3: 56  
    - demand4: 17  
    - demand5: 43  
    - demand6: 62  
    - demand7: 10  
    - demand8: 37  

- **supply_capacity.csv**  
  - $I$ (supplier indices): supplier1, supplier2, supplier3, supplier4, supplier5, supplier6, supplier7, supplier8  
  - $s_i$ values:  
    - supplier1: 60  
    - supplier2: 22  
    - supplier3: 16  
    - supplier4: 14  
    - supplier5: 19  
    - supplier6: 70  
    - supplier7: 60  
    - supplier8: 39  

- **transportation_costs.csv**  
  - $c_{ij}$: cost from supplier $i$ (row "Unnamed: 0") to customer $j$ (column "demandX")  
  - For example, $c_{\text{supply1},\text{demand1}} = 0.03020736643461065$, $c_{\text{supply2},\text{demand2}} = 3.6258726438627473$, etc.  
  - All $c_{ij}$ values are mapped directly from the corresponding CSV row and column identifiers.