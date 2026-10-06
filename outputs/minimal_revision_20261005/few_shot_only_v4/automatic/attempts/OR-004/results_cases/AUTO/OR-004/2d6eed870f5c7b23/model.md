##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i$ to customer group $j$.

- $i \in I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}, \text{S12}\}$
- $j \in J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$

##### Objective Function

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction** (each customer group receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity** (each distribution center does not exceed its supply):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity**:
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- **Sets:**
  - $I$ (distribution centers): S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12
  - $J$ (customer groups): C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12

- **Parameters:**
  - $d_j$ (demand for customer $j$): from "customer_demand.csv", column "demand"
  - $s_i$ (supply capacity for center $i$): from "supply_capacity.csv", column "supply_capacity"
  - $c_{ij}$ (cost per unit from $i$ to $j$): from "transportation_costs.csv", row for $i$, column for $j$

- **Source-column mapping:**
  - "customer_demand.csv": customer $\to$ $j$, demand $\to$ $d_j$
  - "supply_capacity.csv": Unnamed: 0 $\to$ $i$, supply_capacity $\to$ $s_i$
  - "transportation_costs.csv": Unnamed: 0 $\to$ $i$, $Ck$ columns $\to$ $c_{ij}$

---

**Model summary:**  
Minimize $\sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$  
subject to  
$\sum_{i \in I} x_{ij} \geq d_j$ for all $j \in J$  
$\sum_{j \in J} x_{ij} \leq s_i$ for all $i \in I$  
$x_{ij} \geq 0$ for all $i \in I,\, j \in J$  
with all identifiers and coefficients as mapped above.