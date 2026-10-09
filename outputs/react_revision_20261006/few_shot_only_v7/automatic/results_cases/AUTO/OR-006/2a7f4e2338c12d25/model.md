##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$ (warehouses)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$ (stores)

##### Parameters

- $d_j$: demand of store $j$ (from customer_demand.csv)
- $s_i$: supply capacity of warehouse $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from transportation_costs.csv)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity:**  
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- **customer_demand.csv**  
  - $J$ and $d_j$:  
    - C1: $d_{C1} = 45$  
    - C2: $d_{C2} = 23$  
    - C3: $d_{C3} = 94$  
    - C4: $d_{C4} = 92$  
    - C5: $d_{C5} = 57$  
    - C6: $d_{C6} = 52$  
    - C7: $d_{C7} = 23$  
    - C8: $d_{C8} = 99$  
    - C9: $d_{C9} = 99$  
    - C10: $d_{C10} = 77$  

- **supply_capacity.csv**  
  - $I$ and $s_i$:  
    - S1: $s_{S1} = 127$  
    - S2: $s_{S2} = 236$  
    - S3: $s_{S3} = 168$  
    - S4: $s_{S4} = 115$  
    - S5: $s_{S5} = 280$  
    - S6: $s_{S6} = 179$  
    - S7: $s_{S7} = 135$  
    - S8: $s_{S8} = 263$  
    - S9: $s_{S9} = 283$  
    - S10: $s_{S10} = 476$  

- **transportation_costs.csv**  
  - $c_{ij}$:  
    - Rows: S1–S10 (warehouses, in order)  
    - Columns: C1–C10 (stores, in order)  
    - $c_{ij}$ is the value in row $i$, column $j$ of the CSV.  
    - Example: $c_{S1,C1} = 2077.058672521021$, $c_{S1,C2} = 0.0$, ..., $c_{S10,C10} = 0.0$