#### Abstract Mathematical Model

**Index Sets:**
- $I$: set of suppliers (from fixed_cost.csv, transportation_costs.csv)
- $J$: set of branches/customers (from demand.csv, transportation_costs.csv)

**Parameters:**
- $f_i$: fixed cost to open supplier $i \in I$ (from fixed_cost.csv, column "fixed_costs")
- $c_{ij}$: transportation cost per unit from supplier $i \in I$ to branch $j \in J$ (from transportation_costs.csv, columns "C1", ..., "C5")
- $d_j$: demand at branch $j \in J$ (from demand.csv, column "demand")

**Decision Variables:**
- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise
- $x_{ij} \geq 0$: quantity supplied from supplier $i$ to branch $j$

**Objective:**
\[
\min \quad \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

**Constraints:**
1. **Demand Satisfaction:**  
   $\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J$

2. **Supplier Activation:**  
   $x_{ij} \leq d_j y_i \quad \forall i \in I, \forall j \in J$

3. **Variable Domains:**  
   $y_i \in \{0,1\} \quad \forall i \in I$  
   $x_{ij} \geq 0 \quad \forall i \in I, \forall j \in J$

---

#### Data Mapping

- **fixed_cost.csv** (table_id: file_1_view_0):  
  - Supplier index set $I$ from column "Unnamed: 0"
  - Fixed costs $f_i$ from column "fixed_costs"

- **demand.csv** (table_id: file_0_view_0):  
  - Branch index set $J$ from column "customer"
  - Demands $d_j$ from column "demand"

- **transportation_costs.csv** (table_id: file_2_view_0):  
  - Supplier index set $I$ from column "Unnamed: 0"
  - Branch index set $J$ from columns "C1", ..., "C5"
  - Transportation costs $c_{ij}$ from corresponding cells

All index sets, parameters, and relationships are defined according to the exact identifiers and columns in the source files.