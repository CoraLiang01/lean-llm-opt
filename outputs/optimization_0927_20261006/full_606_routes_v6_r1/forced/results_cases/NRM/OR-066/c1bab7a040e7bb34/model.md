#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of suppliers (indexed by $i$), from `file_1_view_0.Unnamed: 0` and `file_2_view_0.Unnamed: 0`
- $J$: Set of supermarkets (indexed by $j$), from `file_0_view_0.customer` and `file_2_view_0` columns

**Parameters:**
- $f_i$: Fixed cost to activate supplier $i$, from `file_1_view_0.fixed_costs`
- $c_{ij}$: Per-unit transportation cost from supplier $i$ to supermarket $j$, from `file_2_view_0` (row: supplier $i$, column: supermarket $j$)
- $d_j$: Demand of supermarket $j$, from `file_0_view_0.demand$

**Decision Variables:**
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise
- $x_{ij} \geq 0$: Amount supplied from supplier $i$ to supermarket $j$

**Objective:**
\[
\min \left( \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \right)
\]

**Constraints:**
1. **Demand Satisfaction:**
   \[
   \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
   \]
2. **Supplier Activation:**
   \[
   \sum_{j \in J} x_{ij} \leq \left( \sum_{j \in J} d_j \right) y_i \quad \forall i \in I
   \]
   (A supplier can only supply if activated; the right-hand side is a valid upper bound since each supermarket must be fully supplied.)

3. **Variable Domains:**
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- $I$ (suppliers): `file_1_view_0.Unnamed: 0` and `file_2_view_0.Unnamed: 0`
- $J$ (supermarkets): `file_0_view_0.customer` and `file_2_view_0` columns (excluding `Unnamed: 0`)
- $f_i$: `file_1_view_0.fixed_costs`
- $c_{ij}$: `file_2_view_0` (row: supplier $i$, column: supermarket $j$)
- $d_j$: `file_0_view_0.demand`

All data used is from the validated, directly returned records of the three files. No additional filtering or aggregation was performed.