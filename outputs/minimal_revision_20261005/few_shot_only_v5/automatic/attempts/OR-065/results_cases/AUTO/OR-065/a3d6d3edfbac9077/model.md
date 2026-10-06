##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether warehouse $i$ is activated (binary).

##### Parameters

- $I = \{\text{S1}, \text{S2}, \text{S3}\}$: set of warehouses.
- $J = \{\text{C1}, \text{C2}, \text{C3}\}$: set of musicians/bands.
- $d_j$: demand of musician/band $j$.
  - $d_{\text{C1}} = 1083$
  - $d_{\text{C2}} = 776$
  - $d_{\text{C3}} = 16214$
- $f_i$: fixed cost for activating warehouse $i$.
  - $f_{\text{S1}} = 102.33$
  - $f_{\text{S2}} = 94.92$
  - $f_{\text{S3}} = 91.83$
- $c_{ij}$: transportation cost per unit from warehouse $i$ to musician/band $j$.
  - $c_{\text{S1},\text{C1}} = 1506.22$, $c_{\text{S1},\text{C2}} = 70.90$, $c_{\text{S1},\text{C3}} = 8.44$
  - $c_{\text{S2},\text{C1}} = 1732.65$, $c_{\text{S2},\text{C2}} = 1780.72$, $c_{\text{S2},\text{C3}} = 567.44$
  - $c_{\text{S3},\text{C1}} = 115.66$, $c_{\text{S3},\text{C2}} = 100.76$, $c_{\text{S3},\text{C3}} = 64.68$
- $M = \sum_{j \in J} d_j = 1083 + 776 + 16214 = 18073$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each musician/band $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
2. **Warehouse activation:**  
   For each warehouse $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   (Inactive warehouses cannot ship goods; $M$ is a valid upper bound.)
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

---

#### Data Mapping

- **Warehouses ($I$):** S1, S2, S3 (from fixed_cost.csv and transportation_costs.csv, column "Unnamed: 0")
- **Musicians/Bands ($J$):** C1, C2, C3 (from demand.csv and transportation_costs.csv, columns "C1", "C2", "C3")
- **Demands ($d_j$):** demand.csv, column "demand"
- **Fixed costs ($f_i$):** fixed_cost.csv, column "fixed_costs"
- **Transportation costs ($c_{ij}$):** transportation_costs.csv, columns "C1", "C2", "C3" per row "Unnamed: 0" (S1, S2, S3)
- **$M$:** sum of all demands

All parameters are mapped directly from the original CSV columns as described.