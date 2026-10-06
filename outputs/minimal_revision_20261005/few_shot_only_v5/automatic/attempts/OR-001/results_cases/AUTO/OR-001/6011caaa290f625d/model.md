##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i$ to customer group $j$.

- $i \in I = \{\text{S1}, \text{S2}, \ldots, \text{S18}\}$
- $j \in J = \{\text{C1}, \text{C2}, \ldots, \text{C18}\}$

##### Objective Function

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
   where $d_j$ is the demand for customer $j$.

2. **Supply capacity:**  
   For each distribution center $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
   where $s_i$ is the supply capacity of center $i$.

3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- **Sets:**
  - $I$ (Distribution Centers): S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17, S18
  - $J$ (Customer Groups): C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18

- **Parameters:**
  - $d_j$: Demand for customer $j$ (from customer_demand.csv, column "demand", indexed by "customer")
  - $s_i$: Supply capacity for center $i$ (from supply_capacity.csv, column "supply_capacity", indexed by "Unnamed: 0")
  - $c_{ij}$: Transportation cost per unit from center $i$ to customer $j$ (from transportation_costs.csv, row "Unnamed: 0" for $i$, column $j$)

- **Variable:**
  - $x_{ij}$: Quantity shipped from $i$ to $j$ (continuous, $\geq 0$)

---

**All identifiers and coefficients are to be taken directly from the source columns as described above.**