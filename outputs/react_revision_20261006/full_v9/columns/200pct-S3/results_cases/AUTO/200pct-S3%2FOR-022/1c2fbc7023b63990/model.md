##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier $i$ is activated (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Branch demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation logic:**  
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   where $M_i = \sum_{j \in J} d_j$ (since there are no explicit supplier capacity limits).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: set of suppliers, from `facility_id` in `file_1_view_0` and `file_2_view_0`
- $J$: set of branches, from `customer_id` in `file_0_view_0` and columns in `file_2_view_0` (after mapping)

##### Parameters and Data Mapping

- $d_j$: demand of branch $j$  
  — Source: `file_0_view_0`, column `demand_units`, indexed by `customer_id`
- $f_i$: fixed opening cost for supplier $i$  
  — Source: `file_1_view_0`, column `fixed_opening_cost`, indexed by `facility_id`
- $c_{ij}$: transportation cost per unit from supplier $i$ to branch $j$  
  — Source: `file_2_view_0`, row `facility_id`, columns `transportation_cost_to_Ck` mapped to $j$
- $M_i$: big-M for each supplier $i$ (set to $\sum_{j \in J} d_j$ for all $i$)

##### Data Mapping

- $I$: All `facility_id` in `file_1_view_0` and `file_2_view_0`
- $J$: All `customer_id` in `file_0_view_0` and columns in `file_2_view_0` mapped as:
    - `transportation_cost_to_C1` $\rightarrow$ `C1`
    - `transportation_cost_to_C2` $\rightarrow$ `C2`
    - `transportation_cost_to_C3` $\rightarrow$ `C3`
    - `transportation_cost_to_C4` $\rightarrow$ `C4`
    - `transportation_cost_to_C5` $\rightarrow$ `C5`
- $d_j$: `file_0_view_0` (`customer_id`, `demand_units`)
- $f_i$: `file_1_view_0` (`facility_id`, `fixed_opening_cost`)
- $c_{ij}$: `file_2_view_0` (`facility_id`, `transportation_cost_to_Ck`)
- $M_i$: $\sum_{j \in J} d_j$ (computed from all $d_j$ in `file_0_view_0`)

No additional constraints or bounds are imposed beyond those above. All index sets and parameters are defined directly from the current CSV data.