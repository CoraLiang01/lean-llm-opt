##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i$ to customer group $j$.

- $i \in I =$ set of distribution centers (from supply_capacity.csv, column "Unnamed: 0")
- $j \in J =$ set of customer groups (from customer_demand.csv, column "customer")

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
   where $d_j$ is the demand for customer $j$ (from customer_demand.csv, column "demand").

2. **Supply capacity:**  
   For each distribution center $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
   where $s_i$ is the supply capacity of center $i$ (from supply_capacity.csv, column "supply_capacity").

3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- **Index sets:**
  - $I =$ all values in supply_capacity.csv, column "Unnamed: 0" (distribution centers):  
    $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17, S18$\}$
  - $J =$ all values in customer_demand.csv, column "customer":  
    $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18$\}$

- **Parameters:**
  - $d_j =$ demand for customer $j$ (from customer_demand.csv, column "demand", table_id: file_0_view_0)
  - $s_i =$ supply capacity for center $i$ (from supply_capacity.csv, column "supply_capacity", table_id: file_1_view_0)
  - $c_{ij} =$ transportation cost per unit from center $i$ to customer $j$ (from transportation_costs.csv, table_id: file_2_view_0, row "Unnamed: 0" = $i$, column $j$)

---

#### Complete Model

**Variables:**  
$x_{ij} \geq 0$ for all $i \in I$, $j \in J$

**Objective:**  
$\displaystyle \min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$

**Subject to:**
- $\displaystyle \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J$
- $\displaystyle \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I$
- $x_{ij} \geq 0 \quad \forall i \in I,\, j \in J$

**Data Mapping:**  
- $I$: file_1_view_0, column "Unnamed: 0"  
- $J$: file_0_view_0, column "customer"  
- $d_j$: file_0_view_0, column "demand"  
- $s_i$: file_1_view_0, column "supply_capacity"  
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" = $i$, column $j$