##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise (binary).

##### Parameters

- $I = \{$S1, S2, ..., S24$\}$: Set of suppliers.
- $J = \{$C1, C2, ..., C25$\}$: Set of supermarkets.
- $d_j$: Demand of supermarket $j \in J$ (from demand.csv).
- $f_i$: Fixed cost of opening supplier $i \in I$ (from fixed_cost.csv).
- $c_{ij}$: Transportation cost per unit from supplier $i$ to supermarket $j$ (from transportation_costs.csv).
- $M = \sum_{j \in J} d_j$: A sufficiently large constant (total demand), used to enforce that inactive suppliers cannot ship.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each supermarket $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Supplier activation:**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
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

- **Suppliers ($I$):** S1, S2, ..., S24 (from fixed_cost.csv and transportation_costs.csv, column "Unnamed: 0")
- **Supermarkets ($J$):** C1, C2, ..., C25 (from demand.csv and transportation_costs.csv, columns "C1"..."C25")
- **Demand ($d_j$):**  
  From demand.csv, column "demand", indexed by "customer" (C1...C25)
- **Fixed costs ($f_i$):**  
  From fixed_cost.csv, column "fixed_costs", indexed by "Unnamed: 0" (S1...S24)
- **Transportation costs ($c_{ij}$):**  
  From transportation_costs.csv, rows indexed by "Unnamed: 0" (S1...S24), columns "C1"..."C25"
- **$M$:** $M = \sum_{j \in J} d_j$ (sum of all demand.csv "demand" values)

All indices and parameter values are to be taken directly from the respective CSV columns as described above.