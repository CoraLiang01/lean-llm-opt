#### Abstract Mathematical Model

**Index Sets:**
- $F$: Set of suppliers (indexed by $i$)
- $S$: Set of stores/customers (indexed by $j$)

**Parameters:**
- $f_i$: Fixed cost to open supplier $i$ (from fixed_cost.csv, column "fixed_costs")
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from transportation_costs.csv, columns indexed by store, rows indexed by supplier)
- $d_j$: Demand at store $j$ (from demand.csv, column "demand")

**Decision Variables:**
- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise
- $x_{ij} \geq 0$: Quantity supplied from supplier $i$ to store $j$

**Objective:**
\[
\min \left( \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in S} c_{ij} x_{ij} \right)
\]

**Constraints:**
1. **Demand Satisfaction:**
   \[
   \sum_{i \in F} x_{ij} = d_j \quad \forall j \in S
   \]
2. **Supplier Activation:**
   \[
   x_{ij} \leq M_{ij} y_i \quad \forall i \in F, \forall j \in S
   \]
   where $M_{ij}$ is a sufficiently large constant (e.g., $M_{ij} \geq d_j$) to allow $x_{ij}$ to be positive only if $y_i = 1$.

3. **Variable Domains:**
   \[
   y_i \in \{0,1\} \quad \forall i \in F
   \]
   \[
   x_{ij} \geq 0 \quad \forall i \in F, \forall j \in S
   \]

---

#### Data Mapping

- **fixed_cost.csv**: 
  - Table ID: file_1_view_0
  - Supplier index set $F$ from column "Unnamed: 0"
  - Fixed cost parameter $f_i$ from column "fixed_costs"

- **transportation_costs.csv**: 
  - Table ID: file_2_view_0
  - Supplier index set $F$ from column "Unnamed: 0"
  - Store index set $S$ from columns "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"
  - Transportation cost parameter $c_{ij}$ from the intersection of supplier row and store column

- **demand.csv**: 
  - Table ID: file_0_view_0
  - Store index set $S$ from column "Customer"
  - Demand parameter $d_j$ from column "demand"

---

**Note:** All data is used as returned by the query, with no additional filtering or aggregation. The model is abstract and symbolic, with all index sets and parameters mapped to their source tables and columns.