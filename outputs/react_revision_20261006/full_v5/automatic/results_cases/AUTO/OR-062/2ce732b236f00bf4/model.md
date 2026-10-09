##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i$ to store $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Store demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation constraint:**  
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ (a valid upper bound since there are no explicit supplier capacity limits).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: Set of suppliers (from fixed_cost.csv and transportation_costs.csv rows)
- $J$: Set of stores (from demand.csv and transportation_costs.csv columns)

##### Data Mapping

- $I$ (Suppliers):  
  - Table: file_1_view_0 (fixed_cost.csv), column: "Unnamed: 0"
  - Table: file_2_view_0 (transportation_costs.csv), row: "Unnamed: 0"
- $J$ (Stores):  
  - Table: file_0_view_0 (demand.csv), column: "Customer"
  - Table: file_2_view_0 (transportation_costs.csv), columns: ["CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"]
- $d_j$:  
  - Table: file_0_view_0 (demand.csv), column: "demand", indexed by "Customer"
- $f_i$:  
  - Table: file_1_view_0 (fixed_cost.csv), column: "fixed_costs", indexed by "Unnamed: 0"
- $c_{ij}$:  
  - Table: file_2_view_0 (transportation_costs.csv), value at row "Unnamed: 0" = $i$, column = $j$
- $M$:  
  - $M = \sum_{j \in J} d_j$ (sum over all store demands from file_0_view_0)

##### Notes

- All parameters and index sets are defined directly from the CSV data as described above.
- The model ensures that each store's demand is exactly met, suppliers are only used if activated, and total cost (fixed + transportation) is minimized.