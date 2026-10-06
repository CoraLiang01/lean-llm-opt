##### Decision Variables

- $x_{ij} \geq 0$: Amount shipped from warehouse $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise (binary).

##### Parameters

- $I$: Set of warehouses, from column "Warehouse (i)" in PotentialWarehouses_Costs.csv.
- $J$: Set of stores, from column "Store (j)" in Stores_Demands.csv.
- $f_i$: Opening cost of warehouse $i$, from column "Opening Cost (fi)" in PotentialWarehouses_Costs.csv.
- $u_i$: Capacity of warehouse $i$, from column "Capacity (units)" in PotentialWarehouses_Costs.csv.
- $d_j$: Demand of store $j$, from column "Demand (units, dj)" in Stores_Demands.csv.
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to store $j$, from TransportationCost.csv, with warehouse $i$ corresponding to row "Unnamed: 1" and store $j$ corresponding to column "Wk" (where $k$ matches warehouse/store indices).

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

- $I$: All values in column "Warehouse (i)" of table_id: file_0_view_0 (PotentialWarehouses_Costs.csv)
- $J$: All values in column "Store (j)" of table_id: file_1_view_0 (Stores_Demands.csv)
- $f_i$: "Opening Cost (fi)" in file_0_view_0, keyed by "Warehouse (i)"
- $u_i$: "Capacity (units)" in file_0_view_0, keyed by "Warehouse (i)"
- $d_j$: "Demand (units, dj)" in file_1_view_0, keyed by "Store (j)"
- $c_{ij}$: Entry in file_2_view_0 (TransportationCost.csv) at row where "Unnamed: 1" = "Wi" and column "Wj", for warehouse $i$ and store $j$ (with $i$, $j$ matching warehouse/store indices as per $I$, $J$)

---

All sets, parameters, and constraints are defined directly from the provided CSV data.