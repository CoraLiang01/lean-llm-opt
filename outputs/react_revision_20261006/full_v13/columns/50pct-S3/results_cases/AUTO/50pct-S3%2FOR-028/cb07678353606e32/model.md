##### Decision Variables

- $y_i \in \{0,1\}$: 1 if warehouse $i \in I$ is opened, 0 otherwise.
- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   (Each store's demand is fully met.)

2. **Warehouse capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i, \quad \forall i \in I
   \]
   (A warehouse cannot ship more than its capacity, and only if it is open.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of warehouses, from column "Warehouse (i)" in PotentialWarehouses_Costs.csv and columns/rows "W1"–"W11" in TransportationCost.csv.
- $J$: set of stores, from column "Store (j)" in Stores_Demands.csv and columns/rows "W1"–"W11" in TransportationCost.csv.
- $f_i$: opening cost of warehouse $i$, from column "Opening Cost (fi)" in PotentialWarehouses_Costs.csv.
- $u_i$: capacity of warehouse $i$, from column "Capacity (units)" in PotentialWarehouses_Costs.csv.
- $d_j$: demand of store $j$, from column "Demand (units, dj)" in Stores_Demands.csv.
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$, from entry $(i,j)$ in TransportationCost.csv, with warehouse and store indices mapped as below.

##### Data Mapping

- $I$ (warehouses): all values in column "Warehouse (i)" of table_id file_0_view_0 (PotentialWarehouses_Costs.csv).
- $J$ (stores): all values in column "Store (j)" of table_id file_1_view_0 (Stores_Demands.csv).
- $f_i$: column "Opening Cost (fi)" in table_id file_0_view_0, keyed by "Warehouse (i)".
- $u_i$: column "Capacity (units)" in table_id file_0_view_0, keyed by "Warehouse (i)".
- $d_j$: column "Demand (units, dj)" in table_id file_1_view_0, keyed by "Store (j)".
- $c_{ij}$: entry in table_id file_2_view_0 (TransportationCost.csv) at row with "Unnamed: 1" = $i$ and column $j$ (where $i$ and $j$ are mapped as "W1"–"W11").

- The mapping between warehouse/store indices and the transportation cost matrix is:
    - Warehouse $i$ in $I$ corresponds to row and column labels "W1"–"W11" in file_2_view_0.
    - Store $j$ in $J$ corresponds to row and column labels "W1"–"W11" in file_2_view_0.

All index sets, parameters, and constraints are defined directly from the current CSV data.