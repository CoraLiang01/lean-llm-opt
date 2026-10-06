##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise (binary).

##### Parameters

- $I$: Set of suppliers, from `file_1_view_0.Unnamed: 3`.
- $J$: Set of stores, from `file_0_view_0.Customer`.
- $d_j$: Demand at store $j$, from `file_0_view_0.demand`.
- $f_i$: Fixed cost to activate supplier $i$, from `file_1_view_0.fixed_costs`.
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$, from `file_2_view_0` (rows: `Unnamed: 4` = supplier $i$, columns: store $j$).

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
   \sum_{j \in J} x_{ij} \leq M \cdot y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ (total demand across all stores).

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Data Mapping

- $I$: All unique values in `file_1_view_0.Unnamed: 3`
- $J$: All unique values in `file_0_view_0.Customer`
- $d_j$: `file_0_view_0.demand` for store $j$
- $f_i$: `file_1_view_0.fixed_costs` for supplier $i$
- $c_{ij}$: Entry in `file_2_view_0` where row `Unnamed: 4` = supplier $i$, column = store $j$
- $M$: $\sum_{j \in J} d_j$ (sum over all `file_0_view_0.demand`)

**All index sets, parameters, and coefficients are to be taken directly from the referenced columns and rows in the CSV files as described above.**