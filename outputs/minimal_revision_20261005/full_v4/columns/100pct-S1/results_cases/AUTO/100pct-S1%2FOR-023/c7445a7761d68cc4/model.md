##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Store demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation logic:**  
   \[
   \sum_{j \in J} x_{ij} \leq M \cdot y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ (total demand; a valid upper bound since no supplier capacity is specified).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \qquad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: Set of suppliers (from `file_1_view_0`, column `Unnamed: 2`)
- $J$: Set of stores (from `file_0_view_0`, column `Customer`)

##### Parameters and Data Mapping

- $d_j$: Demand of store $j$  
  — Source: `file_0_view_0`, columns:  
    - Store index: `Customer`  
    - Demand: `demand`
- $f_i$: Fixed cost for supplier $i$  
  — Source: `file_1_view_0`, columns:  
    - Supplier index: `Unnamed: 2`  
    - Fixed cost: `fixed_costs`
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$  
  — Source: `file_2_view_0`, columns:  
    - Supplier index: `Unnamed: 2` (rows)  
    - Store index: column headers matching store names (e.g., `BANCROFT`, `CLARINDA`, etc.)
- $M$: $\sum_{j \in J} d_j$ (total demand; computed from all $d_j$ in `file_0_view_0`)

##### Data Mapping

- Suppliers $I$:  
  - Table: `file_1_view_0`, column: `Unnamed: 2`
- Stores $J$:  
  - Table: `file_0_view_0`, column: `Customer`
- Demand $d_j$:  
  - Table: `file_0_view_0`, columns: `Customer`, `demand`
- Fixed cost $f_i$:  
  - Table: `file_1_view_0`, columns: `Unnamed: 2`, `fixed_costs`
- Transportation cost $c_{ij}$:  
  - Table: `file_2_view_0`, rows: `Unnamed: 2` (supplier), columns: store names
- $M$:  
  - $M = \sum_{j \in J} d_j$ (sum over all `demand` in `file_0_view_0`)

---

**All index sets, parameters, and coefficients are to be taken directly from the referenced columns in the CSV files as described above.**