##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if plant $i$ is built (opened), 0 otherwise (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   (Each customer's demand must be fully met.)

2. **Plant capacity (only if opened):**  
   \[
   \sum_{j \in J} x_{ij} \leq \text{cap}_i \, y_i, \quad \forall i \in I
   \]
   (A plant can only supply up to its capacity if it is built; if not built, it supplies nothing.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of plants, from column `plant` in `file_0_view_0` (`cost.csv`)
- $J$: set of customers, from column `customer` in `file_1_view_0` (`demand.csv`)
- $f_i$: fixed opening cost of plant $i$, from column `fixed_cost` in `file_0_view_0`
- $\text{cap}_i$: capacity of plant $i$, from column `capacity` in `file_0_view_0`
- $c_{ij}$: per-unit transport cost from plant $i$ to customer $j$, from columns `C1`–`C15` in `file_0_view_0`
- $d_j$: demand of customer $j$, from column `demand` in `file_1_view_0`

##### Data Mapping

- Plants $I$: `file_0_view_0`, column `plant`
- Customers $J$: `file_1_view_0`, column `customer`
- Fixed costs $f_i$: `file_0_view_0`, column `fixed_cost`
- Plant capacities $\text{cap}_i$: `file_0_view_0`, column `capacity`
- Per-unit transport costs $c_{ij}$: `file_0_view_0`, columns `C1`–`C15` (row: plant $i$, column: customer $j$)
- Customer demands $d_j$: `file_1_view_0`, column `demand` (row: customer $j$)

##### Summary

This is a capacitated facility location problem with fixed plant opening costs, per-unit shipping costs, plant capacities, and customer demands, all directly mapped to the provided CSV data.