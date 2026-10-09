##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (operational), 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

where:
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$.
- $f_i$: Fixed cost to activate supplier $i$.

##### Constraints

1. **Demand Satisfaction:**  
   For each store $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
   where $d_j$ is the demand at store $j$.

2. **Supplier Activation:**  
   For each supplier $i \in I$,
   \[
   x_{ij} \geq 0 \quad \forall j \in J
   \]
   (No explicit capacity or activation-conditioned bounds are specified; all $x_{ij}$ are nonnegative and unconstrained except by demand.)

3. **Binary Activation:**  
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Data Mapping

- **Suppliers ($I$):**  
  MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES  
  (from `fixed_cost.csv` Unnamed: 0)

- **Stores ($J$):**  
  CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT  
  (from `transportation_costs.csv` columns, excluding Unnamed: 0)

- **Demands ($d_j$):**  
  - Customer_1: 2397  
  - Customer_2: 1889  
  - Customer_3: 2518  
  - Customer_4: 3218  
  - Customer_5: 1813  
  (from `demand.csv`)

- **Fixed Costs ($f_i$):**  
  - MOUNT AYR: 96.58  
  - WAUKEE: 94.06  
  - WAVERLY: 94.37  
  - PELLA: 82.88  
  - DES MOINES: 94.96  
  (from `fixed_cost.csv`)

- **Transportation Costs ($c_{ij}$):**  
  For each supplier $i$ and store $j$, $c_{ij}$ is given by the corresponding entry in `transportation_costs.csv` (rows: suppliers, columns: stores).

- **Variable Domains:**  
  $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

**Note:**  
- All data identifiers and mappings are preserved as in the source files.
- No supplier capacity or activation-conditioned bounds are imposed unless specified. All $x_{ij}$ are nonnegative and only limited by demand satisfaction.