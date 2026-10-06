##### Decision Variables

- $x_{ij} \geq 0$: Quantity of Adidas products shipped from supplier $i$ to store $j$ (continuous), for all suppliers $i$ and stores $j$.
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise.

##### Parameters

- $f_i$: Fixed cost for opening supplier $i$ (from column "fixed_costs" in table_id "file_1_view_0", row_id_mapping for suppliers).
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from table_id "file_2_view_0", row_id_mapping for suppliers, column_id_mapping for stores).
- $d_j$: Demand for Adidas products at store $j$ (from column "demand" in table_id "file_0_view_0", row_id_mapping for stores).
- $M = \sum_{j} d_j$: A sufficiently large constant, equal to the total demand across all stores.

##### Objective Function

\[
\min \sum_{i} \sum_{j} c_{ij} x_{ij} + \sum_{i} f_i y_i
\]

##### Constraints

1. **Demand satisfaction at each store:**
   \[
   \sum_{i} x_{ij} = d_j, \quad \forall j
   \]
2. **Supplier activation constraint:**
   \[
   \sum_{j} x_{ij} \leq M y_i, \quad \forall i
   \]
3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Data Mapping

- Suppliers $i$: row_id_mapping in table_id "file_1_view_0" and "file_2_view_0" (e.g., "S1", "S2", ..., "S6")
- Stores $j$: column_id_mapping in table_id "file_2_view_0" and row_id_mapping in table_id "file_0_view_0" (e.g., "C1", "C2", ..., "C6")
- $f_i$: "fixed_costs" column, table_id "file_1_view_0", row_id_mapping for $i$
- $c_{ij}$: table_id "file_2_view_0", row_id_mapping for $i$, column_id_mapping for $j$
- $d_j$: "demand" column, table_id "file_0_view_0", row_id_mapping for $j$
- $M = \sum_{j} d_j$: sum of "demand" column in table_id "file_0_view_0"

No additional capacity limits are imposed beyond the activation constraint. All parameters are mapped directly to the supplied CSV data.