#### Index Sets

- $F$: set of suppliers (from file_1_view_0, column "Unnamed: 0" in fixed_cost.csv and file_2_view_0, row "Unnamed: 0" in transportation_costs.csv)
- $C$: set of branches/customers (from file_0_view_0, column "customer" in demand.csv and file_2_view_0, columns "C1", "C2", ... in transportation_costs.csv)

#### Parameters

- $f_i$: fixed cost for opening supplier $i \in F$ (from file_1_view_0, column "fixed_costs")
- $t_{ij}$: transportation cost per unit from supplier $i \in F$ to branch $j \in C$ (from file_2_view_0, entry at row $i$, column $j$)
- $d_j$: demand at branch $j \in C$ (from file_0_view_0, column "demand")

#### Decision Variables

- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise
- $x_{ij} \geq 0$: quantity supplied from supplier $i$ to branch $j$

#### Objective

$$
\min \left( \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in C} t_{ij} x_{ij} \right)
$$

#### Constraints

1. **Demand Satisfaction:**  
   For every branch $j \in C$,
   $$
   \sum_{i \in F} x_{ij} = d_j
   $$
2. **Supplier Activation:**  
   For every supplier $i \in F$ and branch $j \in C$,
   $$
   x_{ij} \leq d_j y_i
   $$
   (No supply from a closed supplier; $d_j$ is a valid upper bound since no branch can receive more than its demand from any supplier.)

3. **Variable Domains:**  
   $$
   y_i \in \{0,1\}, \quad \forall i \in F
   $$
   $$
   x_{ij} \geq 0, \quad \forall i \in F, \forall j \in C
   $$

---

#### Data Mapping

- **file_1_view_0 (fixed_cost.csv):**
  - Supplier index set $F$ from column "Unnamed: 0"
  - Fixed cost parameter $f_i$ from column "fixed_costs"
- **file_0_view_0 (demand.csv):**
  - Branch index set $C$ from column "customer"
  - Demand parameter $d_j$ from column "demand"
- **file_2_view_0 (transportation_costs.csv):**
  - Supplier index set $F$ from row "Unnamed: 0"
  - Branch index set $C$ from columns "C1", "C2", ...
  - Transportation cost parameter $t_{ij}$ from entry at row $i$ (supplier), column $j$ (branch)

All data is used as returned by the query, with no additional filtering or aggregation.