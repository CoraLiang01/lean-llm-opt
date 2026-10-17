#### Index Sets
- $I$: set of suppliers (from file_1_view_0, column "Unnamed: 0")
- $J$: set of branches (from file_0_view_0, column "customer")

#### Parameters
- $f_i$: fixed cost for opening supplier $i \in I$ (from file_1_view_0, column "fixed_costs")
- $c_{ij}$: transportation cost per unit from supplier $i \in I$ to branch $j \in J$ (from file_2_view_0, columns "Unnamed: 0" for supplier, "C1", "C2", "C3", "C4", "C5" for branches)
- $d_j$: demand at branch $j \in J$ (from file_0_view_0, column "demand")

#### Decision Variables
- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise
- $x_{ij} \geq 0$: quantity supplied from supplier $i$ to branch $j$

#### Objective
$$
\min \left( \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \right)
$$

#### Constraints

1. **Demand Satisfaction (for each branch):**
   $$
   \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
   $$

2. **Supplier Activation (linking supply to open suppliers):**
   $$
   \sum_{j \in J} x_{ij} \leq M_i y_i \quad \forall i \in I
   $$
   where $M_i$ is a sufficiently large constant (e.g., $M_i = \sum_{j \in J} d_j$).

3. **Variable Domains:**
   $$
   y_i \in \{0,1\} \quad \forall i \in I
   $$
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- **file_1_view_0 (fixed_cost.csv):**
  - Supplier index set $I$ from column "Unnamed: 0"
  - Fixed cost parameter $f_i$ from column "fixed_costs"
- **file_0_view_0 (demand.csv):**
  - Branch index set $J$ from column "customer"
  - Demand parameter $d_j$ from column "demand"
- **file_2_view_0 (transportation_costs.csv):**
  - Transportation cost parameter $c_{ij}$ from row "Unnamed: 0" (supplier), columns "C1", "C2", "C3", "C4", "C5" (branches $J$)

All data is used as returned by the query, with no additional filtering or aggregation.