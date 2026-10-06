##### Decision Variables

- $x_{ij} \geq 0$: Amount shipped from warehouse $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise (binary).

##### Parameters

- $f_i$: Opening cost for warehouse $i$.
- $u_i$: Capacity of warehouse $i$.
- $d_j$: Demand of store $j$.
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to store $j$.

##### Index Sets

- $I$: Set of warehouses, $I = \{\text{all } i \mid$ "Warehouse (i)" in PotentialWarehouses_Costs.csv$\}$
- $J$: Set of stores, $J = \{\text{all } j \mid$ "Store (j)" in Stores_Demands.csv$\}$

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

- $I$ (warehouses): All "Warehouse (i)" in table_id: file_0_view_0, column: "Warehouse (i)"
- $J$ (stores): All "Store (j)" in table_id: file_1_view_0, column: "Store (j)"
- $f_i$: Opening Cost (fi) from table_id: file_0_view_0, column: "Opening Cost (fi)", keyed by "Warehouse (i)"
- $u_i$: Capacity (units) from table_id: file_0_view_0, column: "Capacity (units)", keyed by "Warehouse (i)"
- $d_j$: Demand (units, dj) from table_id: file_1_view_0, column: "Demand (units, dj)", keyed by "Store (j)"
- $c_{ij}$: Transportation cost from warehouse $i$ to store $j$ from table_id: file_2_view_0, row: "Unnamed: 1" (warehouse $i$), column: $Wk$ (store $j$), where $Wk$ matches warehouse/store indices as per relationships

---

All index sets, parameters, and constraints are defined directly from the provided CSV data.