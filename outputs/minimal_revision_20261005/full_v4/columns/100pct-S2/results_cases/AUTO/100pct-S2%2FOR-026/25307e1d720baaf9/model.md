##### Decision Variables

- $x_{ij} \geq 0$: Amount shipped from plant $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if plant $i$ is built (opened), 0 otherwise (binary).

##### Parameters

- $I$: Set of plants, $I = \{\text{F1}, \ldots, \text{F15}\}$ (from `file_0_view_0`, column `facility_id`)
- $J$: Set of customers, $J = \{\text{C1}, \ldots, \text{C15}\}$ (from `file_1_view_0`, column `customer_id`)
- $f_i$: Fixed opening cost for plant $i$ (from `file_0_view_0`, column `fixed_opening_cost`)
- $K_i$: Capacity of plant $i$ (from `file_0_view_0`, column `facility_capacity`)
- $d_j$: Demand of customer $j$ (from `file_1_view_0`, column `demand_units`)
- $c_{ij}$: Per-unit transportation cost from plant $i$ to customer $j$ (from `file_0_view_0`, columns `transportation_cost_to_C1` ... `transportation_cost_to_C15`)

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Plant capacity:**  
   For each plant $i \in I$,
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

- $I$ (plants): `file_0_view_0`, column `facility_id`
- $J$ (customers): `file_1_view_0`, column `customer_id`
- $f_i$: `file_0_view_0`, column `fixed_opening_cost`, indexed by `facility_id`
- $K_i$: `file_0_view_0`, column `facility_capacity`, indexed by `facility_id`
- $d_j$: `file_1_view_0`, column `demand_units`, indexed by `customer_id`
- $c_{ij}$: `file_0_view_0`, columns `transportation_cost_to_C1` ... `transportation_cost_to_C15`, indexed by `facility_id` and customer $j$ (column suffix matches `customer_id`)

---

**All sets, parameters, and indices are defined exactly as present in the CSV data.**