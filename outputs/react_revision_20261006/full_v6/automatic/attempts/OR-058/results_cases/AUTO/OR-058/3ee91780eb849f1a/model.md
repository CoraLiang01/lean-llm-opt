##### Decision Variables

- $x_{ij} \geq 0$: Quantity of Adidas products shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Store demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation (no shipment from closed suppliers):**  
   \[
   x_{ij} \leq D_j y_i, \quad \forall i \in I, \forall j \in J
   \]
   where $D_j$ is the demand of store $j$.
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: Set of suppliers, from column "Unnamed: 0" in `fixed_cost.csv` and `transportation_costs.csv`.
- $J$: Set of stores, from column "customer" in `demand.csv` and columns in `transportation_costs.csv`.

##### Parameters and Data Mapping

- $d_j$: Demand of store $j$  
  — Source: `demand.csv`, table_id: file_0_view_0, column: "demand", indexed by "customer".
- $f_i$: Fixed cost for supplier $i$  
  — Source: `fixed_cost.csv`, table_id: file_1_view_0, column: "fixed_costs", indexed by "Unnamed: 0".
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$  
  — Source: `transportation_costs.csv`, table_id: file_2_view_0, row: "Unnamed: 0" (supplier), column: store $j$.

##### Data Mapping

- $I$ = all supplier IDs from `fixed_cost.csv`["Unnamed: 0"] and `transportation_costs.csv`["Unnamed: 0"], table_id: file_1_view_0 and file_2_view_0.
- $J$ = all store IDs from `demand.csv`["customer"] and `transportation_costs.csv` columns, table_id: file_0_view_0 and file_2_view_0.
- $d_j$ = `demand.csv`, table_id: file_0_view_0, column: "demand", indexed by "customer".
- $f_i$ = `fixed_cost.csv`, table_id: file_1_view_0, column: "fixed_costs", indexed by "Unnamed: 0".
- $c_{ij}$ = `transportation_costs.csv`, table_id: file_2_view_0, row: "Unnamed: 0" (supplier), column: store $j$.

All index sets and parameters are defined by the full set of entities in the current CSV files. No additional capacity or resource bounds are imposed beyond those described above.