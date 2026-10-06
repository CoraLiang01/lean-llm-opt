##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Supermarket demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation:**  
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j = 3346$.
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Parameters

- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17, S18, S19, S20, S21, S22, S23, S24$\}$
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, C20, C21, C22, C23, C24, C25$\}$

- $d_j$: demand for supermarket $j$ (see mapping below)
- $f_i$: fixed cost for supplier $i$ (see mapping below)
- $c_{ij}$: transportation cost per unit from supplier $i$ to supermarket $j$ (see mapping below)

##### Data Mapping

- **Demand vector $d_j$** (from 'demand.csv', columns: customer, demand):

  | $j$   | $d_j$ |
  |-------|-------|
  | C1    | 1097  |
  | C2    | 61    |
  | C3    | 11    |
  | C4    | 7     |
  | C5    | 82    |
  | C6    | 37    |
  | C7    | 483   |
  | C8    | 582   |
  | C9    | 223   |
  | C10   | 89    |
  | C11   | 60    |
  | C12   | 55    |
  | C13   | 122   |
  | C14   | 66    |
  | C15   | 12    |
  | C16   | 21    |
  | C17   | 53    |
  | C18   | 105   |
  | C19   | 1     |
  | C20   | 253   |
  | C21   | 10    |
  | C22   | 53    |
  | C23   | 24    |
  | C24   | 122   |
  | C25   | 42    |

- **Fixed cost vector $f_i$** (from 'fixed_cost.csv', columns: Unnamed: 0, fixed_costs):

  | $i$   | $f_i$ |
  |-------|-------|
  | S1    | 98.88 |
  | S2    | 99.73 |
  | S3    | 94.01 |
  | S4    | 93.77 |
  | S5    | 107.59|
  | S6    | 112.65|
  | S7    | 97.05 |
  | S8    | 103   |
  | S9    | 90.45 |
  | S10   | 96.73 |
  | S11   | 96.43 |
  | S12   | 112.19|
  | S13   | 102.58|
  | S14   | 88.85 |
  | S15   | 82.57 |
  | S16   | 91.65 |
  | S17   | 101.38|
  | S18   | 102.59|
  | S19   | 105.97|
  | S20   | 85.31 |
  | S21   | 104.52|
  | S22   | 100.2 |
  | S23   | 103.79|
  | S24   | 105.27|

- **Transportation cost matrix $c_{ij}$** (from 'transportation_costs.csv', rows: Unnamed: 0 = $i$, columns: $j$):

  - Each row corresponds to a supplier $i$ (S1 to S24).
  - Each column corresponds to a supermarket $j$ (C1 to C25).
  - $c_{ij}$ is the value at row $i$, column $j$.

  (For brevity, see the original CSV for the full $24 \times 25$ matrix; all values are preserved as in the source.)

- **Big-M parameter:** $M = \sum_{j \in J} d_j = 3346$

---

**Source-Column Data Mapping:**

- demand.csv: customer $\to$ $j$, demand $\to$ $d_j$
- fixed_cost.csv: Unnamed: 0 $\to$ $i$, fixed_costs $\to$ $f_i$
- transportation_costs.csv: Unnamed: 0 $\to$ $i$, C1...C25 $\to$ $c_{ij}$

---

**Sets:**
- Suppliers $I$: S1, S2, ..., S24
- Supermarkets $J$: C1, C2, ..., C25

**Parameters:**
- $d_j$: as above
- $f_i$: as above
- $c_{ij}$: as above

**Variables:**
- $x_{ij} \geq 0$ (continuous)
- $y_i \in \{0,1\}$

**Objective:**
- Minimize total fixed and transportation costs.

**Constraints:**
- Each supermarket's demand is met.
- No shipments from closed suppliers.
- All variables in correct domains.