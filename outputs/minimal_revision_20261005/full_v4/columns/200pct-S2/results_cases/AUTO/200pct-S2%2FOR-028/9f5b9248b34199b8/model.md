##### Decision Variables

- $x_{ij} \geq 0$: Amount shipped from warehouse $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise (binary).

##### Parameters

- $I$: Set of warehouses, from column "Warehouse (i)" in table_id file_0_view_0.
- $J$: Set of stores, from column "Store (j)" in table_id file_1_view_0.
- $f_i$: Opening cost of warehouse $i$, from column "Opening Cost (fi)" in table_id file_0_view_0.
- $u_i$: Capacity of warehouse $i$, from column "Capacity (units)" in table_id file_0_view_0.
- $d_j$: Demand of store $j$, from column "Demand (units, dj)" in table_id file_1_view_0.
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to store $j$, from entry [row $i$, column $j$] in table_id file_2_view_0, with warehouse and store indices mapped as described below.

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Warehouse capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i, \quad \forall i \in I
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

---

#### Data Mapping

- $I$ (warehouses): All values in column "Warehouse (i)" of table_id file_0_view_0.
- $J$ (stores): All values in column "Store (j)" of table_id file_1_view_0.
- $f_i$: For each $i \in I$, value in column "Opening Cost (fi)" of table_id file_0_view_0, row where "Warehouse (i)" = $i$.
- $u_i$: For each $i \in I$, value in column "Capacity (units)" of table_id file_0_view_0, row where "Warehouse (i)" = $i$.
- $d_j$: For each $j \in J$, value in column "Demand (units, dj)" of table_id file_1_view_0, row where "Store (j)" = $j$.
- $c_{ij}$: For each $i \in I$, $j \in J$, value in table_id file_2_view_0 at [row with "Unnamed: 3" = "W$i$", column "W$j$"].

---

**Index mapping for $c_{ij}$:**  
- Warehouse $i$ in $I$ corresponds to row with "Unnamed: 3" = "W$i$" in file_2_view_0.
- Store $j$ in $J$ corresponds to column "W$j$" in file_2_view_0.

---

**All sets, parameters, and coefficients are to be taken directly from the referenced columns and tables as described above.**