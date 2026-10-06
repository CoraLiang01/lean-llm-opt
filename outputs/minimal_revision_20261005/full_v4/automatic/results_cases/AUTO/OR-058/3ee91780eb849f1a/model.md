##### Decision Variables

- $x_{ij} \geq 0$: Quantity of Adidas products shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise (binary).

##### Parameters

- $I$: Set of suppliers, from column "Unnamed: 0" in `fixed_cost.csv` (`file_1_view_0`).
- $J$: Set of stores, from column "customer" in `demand.csv` (`file_0_view_0`).
- $d_j$: Demand of store $j$, from column "demand" in `demand.csv` (`file_0_view_0`).
- $f_i$: Fixed cost for opening supplier $i$, from column "fixed_costs" in `fixed_cost.csv` (`file_1_view_0`).
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$, from entry $(i,j)$ in `transportation_costs.csv` (`file_2_view_0`), with supplier $i$ as row "Unnamed: 0" and store $j$ as column.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction at each store:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation:**
   \[
   \sum_{j \in J} x_{ij} \leq M \cdot y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ (total demand across all stores).

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Data Mapping

- $I$: All values in column "Unnamed: 0" of `fixed_cost.csv` (`file_1_view_0`)
- $J$: All values in column "customer" of `demand.csv` (`file_0_view_0`)
- $d_j$: Column "demand" in `demand.csv` (`file_0_view_0`), indexed by "customer"
- $f_i$: Column "fixed_costs" in `fixed_cost.csv` (`file_1_view_0`), indexed by "Unnamed: 0"
- $c_{ij}$: Entry in `transportation_costs.csv` (`file_2_view_0`), row "Unnamed: 0" = $i$, column = $j$
- $M$: $\sum_{j \in J} d_j$ (sum over all "demand" in `file_0_view_0`)

All index sets and parameters are defined exactly as present in the source CSV files.