##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is activated, 0 otherwise (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   (Each musician/band $j$ receives exactly its demand $d_j$.)

2. **Warehouse activation:**  
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   (No goods can be shipped from warehouse $i$ unless it is activated. $M_i$ is a sufficiently large upper bound, e.g., $M_i = \sum_{j \in J} d_j$.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: Set of warehouses (from fixed_cost.csv, column "Unnamed: 0", table_id: file_1_view_0)
- $J$: Set of musicians/bands (from demand.csv, column "customer", table_id: file_0_view_0)

##### Parameters and Data Mapping

- $d_j$: Demand of musician/band $j$ (from demand.csv, column "demand", table_id: file_0_view_0)
- $f_i$: Fixed cost for warehouse $i$ (from fixed_cost.csv, column "fixed_costs", table_id: file_1_view_0)
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to musician/band $j$ (from transportation_costs.csv, row "Unnamed: 0" for $i$, column $j$, table_id: file_2_view_0)
- $M_i$: Big-M parameter for each warehouse $i$ (set as $M_i = \sum_{j \in J} d_j$ unless otherwise specified)

##### Data Mapping

- $I$: All values in column "Unnamed: 0" of table_id: file_1_view_0 (fixed_cost.csv)
- $J$: All values in column "customer" of table_id: file_0_view_0 (demand.csv)
- $d_j$: Column "demand" in table_id: file_0_view_0, indexed by "customer"
- $f_i$: Column "fixed_costs" in table_id: file_1_view_0, indexed by "Unnamed: 0"
- $c_{ij}$: Table_id: file_2_view_0, row "Unnamed: 0" for $i$, column $j$ (where $j$ matches "customer" in demand.csv)
- $M_i$: $M_i = \sum_{j \in J} d_j$ (sum over all "demand" in table_id: file_0_view_0)

This model determines which warehouses to activate and how much each musician/band should source from each warehouse to minimize total cost, ensuring all demands are met.