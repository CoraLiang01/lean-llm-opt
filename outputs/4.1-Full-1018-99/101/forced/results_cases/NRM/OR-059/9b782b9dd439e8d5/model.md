#### Abstract Mathematical Model

**Index Sets:**
- $S$: set of suppliers (from `fixed_cost.csv` and `transportation_costs.csv` rows)
- $C$: set of dealerships/customers (from `demand.csv` and `transportation_costs.csv` columns)

**Parameters:**
- $f_i$: fixed cost of opening supplier $i \in S$ (from `fixed_cost.csv`, column `fixed_costs`)
- $t_{ij}$: transportation cost per vehicle from supplier $i \in S$ to dealership $j \in C$ (from `transportation_costs.csv`, columns `C1`, ..., `C9`)
- $d_j$: demand (number of vehicles required) at dealership $j \in C$ (from `demand.csv`, column `demand`)

**Decision Variables:**
- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise
- $x_{ij} \geq 0$: number of vehicles supplied from supplier $i$ to dealership $j$

**Objective:**
\[
\min \left( \sum_{i \in S} f_i y_i + \sum_{i \in S} \sum_{j \in C} t_{ij} x_{ij} \right)
\]

**Constraints:**
1. **Demand Satisfaction:**  
   For each dealership $j \in C$,
   \[
   \sum_{i \in S} x_{ij} = d_j
   \]
2. **Supplier Activation:**  
   For all $i \in S$, $j \in C$,
   \[
   x_{ij} \leq d_j y_i
   \]
   (A supplier can only supply if it is open; $d_j$ is a valid upper bound since no dealership can receive more than its demand from any supplier.)

3. **Variable Domains:**  
   \[
   y_i \in \{0,1\}, \quad \forall i \in S
   \]
   \[
   x_{ij} \geq 0, \quad \forall i \in S,\, j \in C
   \]

---

#### Data Mapping

- **Supplier Set $S$ and Fixed Costs $f_i$:**  
  - Table: `fixed_cost.csv` (`file_1_view_0`)  
  - Supplier index: column `Unnamed: 0`  
  - Fixed cost: column `fixed_costs`

- **Dealership Set $C$ and Demands $d_j$:**  
  - Table: `demand.csv` (`file_0_view_0`)  
  - Dealership index: column `customer`  
  - Demand: column `demand`

- **Transportation Costs $t_{ij}$:**  
  - Table: `transportation_costs.csv` (`file_2_view_0`)  
  - Supplier index: row `Unnamed: 0`  
  - Dealership index: columns `C1`, `C2`, ..., `C9`  
  - Transportation cost: value at row (supplier $i$), column (dealership $j$)

---

**Note:**  
All index sets, parameters, and mappings are defined exactly as in the source files and columns returned by CSVQA. No additional constraints or data transformations have been introduced.