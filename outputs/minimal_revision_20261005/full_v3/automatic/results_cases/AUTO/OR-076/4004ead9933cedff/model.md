##### Decision Variables

- $x_{ij} \geq 0$: Amount shipped from warehouse $i$ to customer $j$ (continuous), for all warehouses $i$ and customers $j$.
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise.

##### Parameters

- $c_{ij}$: Unit transportation cost from warehouse $i$ to customer $j$ (from file_0_view_0, columns "Warehouse ID", customer columns; row_id_mapping and column_id_mapping as below).
- $f_i$: Fixed annual opening cost for warehouse $i$ (from file_1_view_0, columns "Warehouse ID", "Fixed_Cost").
- $u_i$: Maximum service capacity of warehouse $i$ (from file_1_view_0, columns "Warehouse ID", "Capacity").
- $d_j$: Demand of customer $j$ (from file_2_view_0, columns "Customer ID", "Demand").

##### Sets

- $I$: Set of warehouses (row_id_mapping from file_0_view_0 and file_1_view_0).
- $J$: Set of customers (column_id_mapping from file_0_view_0 and file_2_view_0).

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   For every customer $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Warehouse capacity:**  
   For every warehouse $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

---

#### Data Mapping

- $c_{ij}$: file_0_view_0, row_id_mapping (Warehouse ID), column_id_mapping (Customer columns C1–C20)
- $f_i$: file_1_view_0, columns "Warehouse ID", "Fixed_Cost"
- $u_i$: file_1_view_0, columns "Warehouse ID", "Capacity"
- $d_j$: file_2_view_0, columns "Customer ID", "Demand"
- $I$: row_id_mapping from file_0_view_0 and file_1_view_0 ("Warehouse ID")
- $J$: column_id_mapping from file_0_view_0 and file_2_view_0 ("Customer ID")