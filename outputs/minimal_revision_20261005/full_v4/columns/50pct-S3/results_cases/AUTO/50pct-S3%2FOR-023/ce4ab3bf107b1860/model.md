##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i$ to store $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise (binary).

##### Sets

- $I$: Set of suppliers, indexed by $i$ (from `file_1_view_0.Unnamed: 1` and `file_2_view_0.Unnamed: 0`)
- $J$: Set of stores, indexed by $j$ (from `file_0_view_0.Customer` and `file_2_view_0` column headers)

##### Parameters

- $d_j$: Demand at store $j$ (from `file_0_view_0.demand`)
- $f_i$: Fixed cost to activate supplier $i$ (from `file_1_view_0.fixed_costs`)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from `file_2_view_0`, row `Unnamed: 0` = $i$, column $j$)
- $M$: A sufficiently large constant, e.g., $M = \sum_{j \in J} d_j$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction at each store:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation logic:**
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

---

#### Data Mapping

- $I$ (suppliers): All unique values in `file_1_view_0.Unnamed: 1` and `file_2_view_0.Unnamed: 0`
- $J$ (stores): All unique values in `file_0_view_0.Customer` and all column names in `file_2_view_0` except `Unnamed: 0` and columns with "previous_period" or "service_status" in their name
- $d_j$: `file_0_view_0.demand` where `Customer` = $j$
- $f_i$: `file_1_view_0.fixed_costs` where `Unnamed: 1` = $i$
- $c_{ij}$: `file_2_view_0` value at row where `Unnamed: 0` = $i$, column $j$
- $M$: $\sum_{j \in J} d_j$ (sum over all `file_0_view_0.demand`)

---

**Note:** All index sets, parameters, and mappings are defined directly from the CSV data as described above. No values are enumerated; all mappings are symbolic and reference the exact source columns and table IDs.