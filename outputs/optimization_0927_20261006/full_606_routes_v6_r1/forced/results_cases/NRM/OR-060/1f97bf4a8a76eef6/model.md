#### Abstract Mathematical Model

**Index Sets:**
- $F$: set of suppliers (indexed by $i$), from `file_1_view_0.Unnamed: 0` and `file_2_view_0.Unnamed: 0`
- $C$: set of supermarkets/customers (indexed by $j$), from `file_0_view_0.customer` and `file_2_view_0` column headers (excluding `Unnamed: 0`)

**Parameters:**
- $f_i$: fixed cost to open supplier $i \in F$ (`file_1_view_0.fixed_costs`)
- $t_{ij}$: transportation cost per unit from supplier $i \in F$ to supermarket $j \in C$ (`file_2_view_0`, entry at row $i$, column $j$)
- $d_j$: demand of supermarket $j \in C$ (`file_0_view_0.demand`)

**Decision Variables:**
- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise
- $x_{ij} \geq 0$: quantity supplied from supplier $i$ to supermarket $j$

**Objective:**
\[
\min \left( \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in C} t_{ij} x_{ij} \right)
\]

**Constraints:**
1. **Demand Satisfaction:**  
   For all $j \in C$,
   \[
   \sum_{i \in F} x_{ij} = d_j
   \]
2. **Supplier Activation:**  
   For all $i \in F$, $j \in C$,
   \[
   x_{ij} \leq d_j y_i
   \]
   (A supplier can only supply if it is open; $d_j$ is an upper bound since no supermarket can receive more than its demand from any one supplier.)

3. **Variable Domains:**  
   \[
   y_i \in \{0,1\} \quad \forall i \in F
   \]
   \[
   x_{ij} \geq 0 \quad \forall i \in F,\, j \in C
   \]

---

#### Data Mapping

- **Supplier Set $F$ and Fixed Costs $f_i$:**  
  - Table: `file_1_view_0` (from "fixed_cost.csv")  
    - Supplier index: `Unnamed: 0`  
    - Fixed cost: `fixed_costs`

- **Supermarket Set $C$ and Demands $d_j$:**  
  - Table: `file_0_view_0` (from "demand.csv")  
    - Supermarket index: `customer`  
    - Demand: `demand`

- **Transportation Costs $t_{ij}$:**  
  - Table: `file_2_view_0` (from "transportation_costs.csv")  
    - Supplier index: `Unnamed: 0` (rows)  
    - Supermarket index: column headers (excluding `Unnamed: 0`)  
    - Transportation cost: cell at row $i$, column $j$

- **All data used is the full set of records from each file, as returned by CSVQA.**