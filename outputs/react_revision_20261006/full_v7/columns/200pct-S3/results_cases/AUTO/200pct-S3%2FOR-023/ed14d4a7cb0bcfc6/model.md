##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Store demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation logic:**  
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   where $M_i$ is a sufficiently large constant (e.g., $M_i = \sum_{j \in J} d_j$) to ensure that if $y_i = 0$, then $x_{ij} = 0$ for all $j$.
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers (from `fixed_cost.csv` and `transportation_costs.csv` rows),  
  $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- $J$: Set of stores (from `demand.csv` and `transportation_costs.csv` columns),  
  $J = \{\text{Customer\_1}, \text{Customer\_2}, \text{Customer\_3}, \text{Customer\_4}, \text{Customer\_5}\}$
- $d_j$: Demand for store $j$ (from `demand.csv`, column `demand`, indexed by `Customer`)
- $f_i$: Fixed cost for supplier $i$ (from `fixed_cost.csv`, column `fixed_costs`, indexed by `Unnamed: 3`)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from `transportation_costs.csv`, row `Unnamed: 4` for supplier, column for store)
- $M_i$: Big-M constant for each supplier $i$ (set as $M_i = \sum_{j \in J} d_j$)

##### Data Mapping

- $I$ (suppliers):  
  - Table: `file_1_view_0` (`fixed_cost.csv`), column: `Unnamed: 3`  
  - Table: `file_2_view_0` (`transportation_costs.csv`), column: `Unnamed: 4`
- $J$ (stores):  
  - Table: `file_0_view_0` (`demand.csv`), column: `Customer`
- $d_j$:  
  - Table: `file_0_view_0` (`demand.csv`), columns: `Customer`, `demand`
- $f_i$:  
  - Table: `file_1_view_0` (`fixed_cost.csv`), columns: `Unnamed: 3`, `fixed_costs`
- $c_{ij}$:  
  - Table: `file_2_view_0` (`transportation_costs.csv`), rows: `Unnamed: 4` (supplier), columns: store names (e.g., `BANCROFT`, `CLARINDA`, etc.)
- $M_i$:  
  - $M_i = \sum_{j \in J} d_j$ (sum over all $d_j$ from `demand.csv`)

**Note:** The mapping between store names in `demand.csv` (`Customer_1`, ...) and columns in `transportation_costs.csv` (e.g., `BANCROFT`, `CLARINDA`, ...) must be established for implementation; for the symbolic model, use the index sets as defined above.