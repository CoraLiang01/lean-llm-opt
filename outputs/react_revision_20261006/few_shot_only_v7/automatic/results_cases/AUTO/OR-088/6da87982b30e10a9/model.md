##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if plant $i$ is built (opened), 0 otherwise.

##### Parameters

- $I = \{\text{F1}, \text{F2}, \ldots, \text{F15}\}$: set of plants.
- $J = \{\text{C1}, \text{C2}, \ldots, \text{C15}\}$: set of customers.
- $f_i$: fixed opening cost for plant $i$ (from cost.csv, column "fixed_cost").
- $u_i$: capacity of plant $i$ (from cost.csv, column "capacity").
- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$ (from cost.csv, columns "C1"–"C15").
- $d_j$: demand of customer $j$ (from demand.csv, column "demand").

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
   \sum_{j \in J} x_{ij} \leq u_i
   \]

3. **Activation constraint:**  
   For each plant $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i
   \]
   (i.e., if $y_i = 0$, then $x_{ij} = 0$ for all $j$; if $y_i = 1$, plant $i$ can supply up to its capacity.)

4. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

---

#### Data Mapping

- **Plants ($I$):** Names from cost.csv, column "plant": F1, F2, ..., F15.
- **Customers ($J$):** Names from demand.csv, column "customer": C1, C2, ..., C15.
- **Fixed opening cost ($f_i$):** cost.csv, column "fixed_cost", for each plant.
- **Plant capacity ($u_i$):** cost.csv, column "capacity", for each plant.
- **Transportation cost ($c_{ij}$):** cost.csv, columns "C1"–"C15", for each plant–customer pair.
- **Customer demand ($d_j$):** demand.csv, column "demand", for each customer.

All indices, parameters, and mappings are directly from the supplied CSV columns and rows.