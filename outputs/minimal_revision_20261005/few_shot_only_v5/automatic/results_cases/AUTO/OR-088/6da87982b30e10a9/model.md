##### Decision Variables

- $x_{ij} \geq 0$: Amount shipped from plant $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if plant $i$ is built (opened), 0 otherwise (binary).

##### Parameters

- $I = \{\text{F1}, \text{F2}, \ldots, \text{F15}\}$: Set of plants.
- $J = \{\text{C1}, \text{C2}, \ldots, \text{C15}\}$: Set of customers.
- $f_i$: Fixed opening cost for plant $i$.
- $K_i$: Capacity of plant $i$.
- $c_{ij}$: Per-unit transportation cost from plant $i$ to customer $j$.
- $d_j$: Demand of customer $j$.

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

- **Plants ($I$):** F1, F2, ..., F15 (from `cost.csv`, column `plant`)
- **Customers ($J$):** C1, C2, ..., C15 (from `cost.csv` columns C1–C15 and `demand.csv` column `customer`)
- **Fixed opening cost ($f_i$):** `cost.csv`, column `fixed_cost`
- **Plant capacity ($K_i$):** `cost.csv`, column `capacity`
- **Per-unit transport cost ($c_{ij}$):** `cost.csv`, columns C1–C15 for each plant row
- **Customer demand ($d_j$):** `demand.csv`, column `demand` for each customer

All parameters are to be taken directly from the corresponding columns and rows of the provided CSV files, preserving all identifiers and values.