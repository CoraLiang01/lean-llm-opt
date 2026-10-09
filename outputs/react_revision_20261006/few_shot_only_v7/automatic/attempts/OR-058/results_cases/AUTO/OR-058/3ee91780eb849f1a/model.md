##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

where:
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$.
- $f_i$: Fixed cost to activate supplier $i$.

##### Constraints

1. **Demand satisfaction at each store:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   where $d_j$ is the demand at store $j$.

2. **Nonnegativity and domains:**
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

3. **(No explicit capacity or activation-conditioned bounds are imposed, as per the problem statement.)**

---

##### Data Mapping

- **Suppliers ($I$):** S1, S2, S3, S4, S5, S6  
  (from `fixed_cost.csv` and `transportation_costs.csv`, column "Unnamed: 0")

- **Stores ($J$):** C1, C2, C3, C4, C5, C6  
  (from `demand.csv` and `transportation_costs.csv`, columns "C1"..."C6")

- **Demands ($d_j$):**  
  - C1: 216  
  - C2: 216  
  - C3: 216  
  - C4: 144  
  - C5: 144  
  - C6: 144  
  (from `demand.csv`, columns "customer", "demand")

- **Fixed costs ($f_i$):**  
  - S1: 98.88  
  - S2: 99.73  
  - S3: 94.01  
  - S4: 93.77  
  - S5: 107.59  
  - S6: 112.65  
  (from `fixed_cost.csv`, columns "Unnamed: 0", "fixed_costs")

- **Transportation costs ($c_{ij}$):**  
  $c_{ij}$ is the entry in `transportation_costs.csv` at row for supplier $i$ and column for store $j$.

  (from `transportation_costs.csv`, rows "Unnamed: 0" = S1...S6, columns "C1"..."C6")

---

**Note:**  
- All parameters and indices are mapped directly from the provided CSV columns and rows.
- There are no supplier capacity or activation-conditioned bounds unless specified in the problem. All $x_{ij}$ are nonnegative and unconstrained above except by demand satisfaction.