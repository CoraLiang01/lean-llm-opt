#### Abstract Mathematical Model

**Index Sets:**
- $F$: set of suppliers (from `fixed_cost.csv`, column `Unnamed: 0`)
- $S$: set of supermarkets (from `demand.csv`, column `customer` and `transportation_costs.csv` columns)

**Parameters:**
- $f_i$: fixed cost of opening supplier $i \in F$ (from `fixed_cost.csv`, column `fixed_costs`)
- $c_{ij}$: transportation cost per unit from supplier $i \in F$ to supermarket $j \in S$ (from `transportation_costs.csv`, entry at row $i$, column $j$)
- $d_j$: demand of supermarket $j \in S$ (from `demand.csv`, column `demand`)

**Decision Variables:**
- $y_i \in \{0,1\}$: $1$ if supplier $i$ is open, $0$ otherwise, for all $i \in F$
- $x_{ij} \geq 0$: quantity supplied from supplier $i$ to supermarket $j$, for all $i \in F$, $j \in S$

**Objective:**
\[
\min \left( \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in S} c_{ij} x_{ij} \right)
\]

**Constraints:**
1. **Demand Satisfaction:**  
   For all $j \in S$,
   \[
   \sum_{i \in F} x_{ij} = d_j
   \]
2. **Supplier Activation:**  
   For all $i \in F$, $j \in S$,
   \[
   x_{ij} \leq d_j y_i
   \]
3. **Variable Domains:**  
   \[
   y_i \in \{0,1\}, \quad \forall i \in F
   \]
   \[
   x_{ij} \geq 0, \quad \forall i \in F,\, j \in S
   \]

---

**Data Mapping:**

- Table `file_1_view_0` (`fixed_cost.csv`):  
  - Supplier index set $F$ from column `Unnamed: 0`  
  - Fixed cost $f_i$ from column `fixed_costs`
- Table `file_0_view_0` (`demand.csv`):  
  - Supermarket index set $S$ from column `customer`  
  - Demand $d_j$ from column `demand`
- Table `file_2_view_0` (`transportation_costs.csv`):  
  - Supplier index set $F$ from column `Unnamed: 0` (rows)  
  - Supermarket index set $S$ from columns `C1`, `C2`, ..., `C25`  
  - Transportation cost $c_{ij}$ from entry at row $i$, column $j$

---

**Notes:**  
- All sets, parameters, and variables are defined symbolically and mapped to their source columns and tables.
- The model minimizes the sum of fixed and transportation costs, ensuring all supermarket demands are met and only open suppliers can supply goods.