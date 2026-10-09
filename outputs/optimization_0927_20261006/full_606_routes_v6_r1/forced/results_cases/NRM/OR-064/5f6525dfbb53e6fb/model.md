#### Abstract Mathematical Model

**Index Sets:**
- $F$: Set of suppliers (indexed by $i$), from `file_1_view_0.Unnamed: 0` and `file_2_view_0.Unnamed: 0`
- $C$: Set of supermarkets (indexed by $j$), from `file_0_view_0.customer` and `file_2_view_0` columns

**Parameters:**
- $f_i$: Fixed cost to open supplier $i \in F$ (`file_1_view_0.fixed_costs`)
- $t_{ij}$: Transportation cost per unit from supplier $i \in F$ to supermarket $j \in C$ (`file_2_view_0`, entry at row $i$, column $j$)
- $d_j$: Demand of supermarket $j \in C$ (`file_0_view_0.demand`)

**Decision Variables:**
- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise
- $x_{ij} \geq 0$: Quantity supplied from supplier $i$ to supermarket $j$

**Objective:**
\[
\min \left( \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in C} t_{ij} x_{ij} \right)
\]

**Constraints:**

1. **Demand Satisfaction:**
   \[
   \sum_{i \in F} x_{ij} = d_j \qquad \forall j \in C
   \]

2. **Supplier Activation:**
   \[
   x_{ij} \leq d_j y_i \qquad \forall i \in F, \forall j \in C
   \]
   (A supplier can only supply if it is open; $d_j$ is an upper bound for $x_{ij}$.)

3. **Variable Domains:**
   \[
   y_i \in \{0,1\} \qquad \forall i \in F
   \]
   \[
   x_{ij} \geq 0 \qquad \forall i \in F, \forall j \in C
   \]

---

#### Data Mapping

- **Supplier Set $F$ and Fixed Costs $f_i$:**  
  - Table: `file_1_view_0` (fixed_cost.csv)  
  - Columns: `Unnamed: 0` (supplier ID), `fixed_costs` (fixed cost per supplier)

- **Supermarket Set $C$ and Demands $d_j$:**  
  - Table: `file_0_view_0` (demand.csv)  
  - Columns: `customer` (supermarket ID), `demand` (demand per supermarket)

- **Transportation Costs $t_{ij}$:**  
  - Table: `file_2_view_0` (transportation_costs.csv)  
  - Row index: `Unnamed: 0` (supplier ID, matches $F$)  
  - Column index: supermarket IDs (matches $C$)  
  - Entries: transportation cost per unit from supplier $i$ to supermarket $j$

---

**Note:**  
- All sets, parameters, and relationships are defined directly from the returned tables and columns, with no additional filtering or aggregation.
- The model ensures all supermarket demands are met, suppliers are only used if open, and total cost (fixed + transportation) is minimized.