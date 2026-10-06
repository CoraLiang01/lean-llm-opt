##### Decision Variables

- $x_{ij} \geq 0$: quantity of goods shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction (each supermarket's demand must be met):**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation (no shipments from closed suppliers):**
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$.

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \qquad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: set of suppliers, from `file_1_view_0`, column `Unnamed: 0`
- $J$: set of supermarkets, from `file_0_view_0`, column `customer`

##### Parameters and Data Mapping

- $d_j$: demand of supermarket $j$, from `file_0_view_0`, columns: `customer`, `demand`
- $f_i$: fixed cost for opening supplier $i$, from `file_1_view_0`, columns: `Unnamed: 0`, `fixed_costs`
- $c_{ij}$: transportation cost per unit from supplier $i$ to supermarket $j$, from `file_2_view_0`, rows indexed by `Unnamed: 0` (supplier), columns indexed by supermarket IDs (`C1`, ..., `C25`)
- $M = \sum_{j \in J} d_j$ (total demand, computed from all $d_j$)

##### Data Mapping

- $I$: `file_1_view_0`, column `Unnamed: 0`
- $J$: `file_0_view_0`, column `customer`
- $d_j$: `file_0_view_0`, columns `customer`, `demand`
- $f_i$: `file_1_view_0`, columns `Unnamed: 0`, `fixed_costs`
- $c_{ij}$: `file_2_view_0`, row `Unnamed: 0` (supplier), column $j$ (supermarket)
- $M$: $\sum_{j \in J} d_j$ (from all $d_j$ in `file_0_view_0`)

##### Notes

- All suppliers and supermarkets present in the data are included in $I$ and $J$.
- The model ensures that only open suppliers ($y_i=1$) can ship goods.
- There are no explicit supplier capacity limits; $M$ is a valid upper bound for constraint 2.