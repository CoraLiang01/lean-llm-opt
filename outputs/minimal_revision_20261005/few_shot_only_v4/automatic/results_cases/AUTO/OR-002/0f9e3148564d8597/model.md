##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from store $i$ to customer group $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}\}$ (stores)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$ (customer groups)

##### Parameters

- $d_j$: demand of customer group $j$ (from customer_demand.csv)
- $s_i$: supply capacity of store $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from store $i$ to customer group $j$ (from transportation_costs.csv)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each customer group $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
2. **Supply capacity:**  
   For each store $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $d_j$:  
  - C1: 11  
  - C2: 1148  
  - C3: 54  
  - C4: 833  
  - C5: 154  
  - C6: 551  
  - C7: 7081  
  - C8: 76  
  - C9: 66  
  - C10: 174  
  - C11: 15  
  - C12: 680  

- $s_i$:  
  - S1: 4  
  - S2: 575  
  - S3: 1504  
  - S4: 178  
  - S5: 228  
  - S6: 50  
  - S7: 3  
  - S8: 6148  
  - S9: 6  
  - S10: 10673  
  - S11: 174  

- $c_{ij}$:  
  - Table: rows = S1...S11, columns = C1...C12, as in transportation_costs.csv (source order preserved; see original data for all coefficients).

---

**All identifiers and coefficients are preserved as in the source files.**