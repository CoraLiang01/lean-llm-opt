##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supply center $i \in I$ to customer group $j \in J$.

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}, \text{S12}\}$
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$

##### Parameters

- $d_j$: demand of customer group $j \in J$
- $s_i$: supply capacity of supply center $i \in I$
- $c_{ij}$: transportation cost per unit from supply center $i$ to customer group $j$

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

- $i$ (supply centers):  
  S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12  
  (from supply_capacity.csv, column "Unnamed: 0")

- $j$ (customer groups):  
  C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12  
  (from customer_demand.csv, column "customer")

- $d_j$ (demand):  
  from customer_demand.csv, column "demand", indexed by "customer"

- $s_i$ (supply capacity):  
  from supply_capacity.csv, column "supply_capacity", indexed by "Unnamed: 0"

- $c_{ij}$ (transportation cost):  
  from transportation_costs.csv, row "Unnamed: 0" = $i$, column $j$

All indices and coefficients are to be used exactly as in the source files and columns.