##### Decision Variables

- $x_{ij} \geq 0$: Amount shipped from facility $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if facility $i$ is opened, 0 otherwise (binary).

##### Parameters

- $I$: Set of facilities (plants), from column `facility_id` in `cost.csv`.
- $J$: Set of customers, from column `customer_id` in `demand.csv`.
- $f_i$: Fixed opening cost for facility $i$, from column `fixed_opening_cost` in `cost.csv`.
- $K_i$: Capacity of facility $i$, from column `facility_capacity` in `cost.csv`.
- $c_{ij}$: Per-unit transportation cost from facility $i$ to customer $j$, from columns `transportation_cost_to_C1` ... `transportation_cost_to_C15` in `cost.csv`.
- $d_j$: Demand of customer $j$, from column `demand_units` in `demand.csv`.

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   For all $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Facility capacity:**  
   For all $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq K_i y_i
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

---

#### Data Mapping

- $I$: All values in `facility_id` column of `cost.csv` (`file_0_view_0`, column `facility_id`)
- $J$: All values in `customer_id` column of `demand.csv` (`file_1_view_0`, column `customer_id`)
- $f_i$: `fixed_opening_cost` column in `cost.csv` (`file_0_view_0`, column `fixed_opening_cost`)
- $K_i$: `facility_capacity` column in `cost.csv` (`file_0_view_0`, column `facility_capacity`)
- $c_{ij}$: `transportation_cost_to_Ck` columns in `cost.csv` (`file_0_view_0`, columns `transportation_cost_to_C1` ... `transportation_cost_to_C15`), where $j$ matches customer $Ck$
- $d_j$: `demand_units` column in `demand.csv` (`file_1_view_0`, column `demand_units`)