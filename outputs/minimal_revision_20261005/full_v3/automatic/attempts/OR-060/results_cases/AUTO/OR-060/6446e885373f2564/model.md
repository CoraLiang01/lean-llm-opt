##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier $i$ to supermarket $j$ (continuous), for all suppliers $i$ and supermarkets $j$.
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise.

##### Parameters

- $f_i$: Fixed cost for opening supplier $i$ (from column "fixed_costs" in table_id: file_1_view_0, row_id: $i$).
- $c_{ij}$: Transportation cost per unit from supplier $i$ to supermarket $j$ (from table_id: file_2_view_0, row_id: $i$, column_id: $j$).
- $d_j$: Demand of supermarket $j$ (from column "demand" in table_id: file_0_view_0, row_id: $j$).

##### Sets

- $I$: Set of suppliers (row_id in file_1_view_0 and file_2_view_0).
- $J$: Set of supermarkets (row_id in file_0_view_0 and column_id in file_2_view_0).

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   For each supermarket $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
2. **Supplier activation:**  
   For each supplier $i \in I$ and supermarket $j \in J$,
   \[
   x_{ij} \leq d_j y_i
   \]
   (A supplier can only supply to a supermarket if it is open; $d_j$ is a valid upper bound for $x_{ij}$ since no supermarket can receive more than its demand from any one supplier.)
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

---

#### Data Mapping

- $f_i$: file_1_view_0, column "fixed_costs", row_id $i$ (supplier).
- $c_{ij}$: file_2_view_0, row_id $i$ (supplier), column_id $j$ (supermarket).
- $d_j$: file_0_view_0, column "demand", row_id $j$ (supermarket).
- $I$: row_id in file_1_view_0 and file_2_view_0 ("S1", ..., "S12").
- $J$: row_id in file_0_view_0 and column_id in file_2_view_0 ("C1", ..., "C12").