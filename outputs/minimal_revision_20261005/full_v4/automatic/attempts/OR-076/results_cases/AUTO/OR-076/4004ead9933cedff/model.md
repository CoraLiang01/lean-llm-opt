##### Decision Variables

- $x_{ij} \geq 0$: Amount shipped from warehouse $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise.

##### Parameters

- $I$: Set of warehouses (from column "Warehouse ID" in table_id: file_1_view_0).
- $J$: Set of customers (from column "Customer ID" in table_id: file_2_view_0).
- $f_i$: Fixed opening cost for warehouse $i$ (from column "Fixed_Cost" in table_id: file_1_view_0).
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to customer $j$ (from table_id: file_0_view_0, row "Warehouse ID", columns $J$).
- $d_j$: Demand of customer $j$ (from column "Demand" in table_id: file_2_view_0).
- $u_i$: Capacity of warehouse $i$ (from column "Capacity" in table_id: file_1_view_0).

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   $\displaystyle \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J$

2. **Warehouse capacity:**  
   $\displaystyle \sum_{j \in J} x_{ij} \leq u_i y_i \quad \forall i \in I$

3. **Variable domains:**  
   $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

---

#### Data Mapping

- $I$: All values in column "Warehouse ID" of table_id: file_1_view_0
- $J$: All values in column "Customer ID" of table_id: file_2_view_0
- $f_i$: "Fixed_Cost" column in table_id: file_1_view_0, indexed by "Warehouse ID"
- $u_i$: "Capacity" column in table_id: file_1_view_0, indexed by "Warehouse ID"
- $d_j$: "Demand" column in table_id: file_2_view_0, indexed by "Customer ID"
- $c_{ij}$: Entry in table_id: file_0_view_0, row "Warehouse ID" = $i$, column $j$ (customer ID)

---

**Index sets $I$ and $J$ are defined by all warehouse and customer IDs present in the respective tables. All parameters are mapped directly to their respective columns as specified above.**