#### Symbolic Mathematical Model

Let:
- $I$ = set of vehicle types (indexed by $i$), as given by all ProductName in products.csv.
- $x_i$ = number of units of vehicle type $i$ to order daily (decision variable), $x_i \in \mathbb{Z}_{\geq 0}$.
- $v_i$ = benefit coefficient for vehicle type $i$ (from Value column).
- $w_i$ = inventory weight (space requirement) for vehicle type $i$ (from Weight column).
- $C$ = total inventory capacity (from Capacity column in capacity.csv).

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

#### Data Mapping

- $I$: All records in file_1_view_0 (products.csv), column ProductName.
- $v_i$: file_1_view_0 (products.csv), column Value, keyed by ProductName.
- $w_i$: file_1_view_0 (products.csv), column Weight, keyed by ProductName.
- $C$: file_0_view_0 (capacity.csv), column Capacity (single value).

**Variables:**
- $x_i$: Number of units of vehicle type $i$ to order daily, integer, $\geq 0$, for each $i \in I$.

**Objective:**
- Maximize total benefit: $\sum_{i \in I} v_i x_i$

**Constraint:**
- Total inventory weight does not exceed capacity: $\sum_{i \in I} w_i x_i \leq C$ (with $C$ from file_0_view_0, Capacity)

**Domain:**
- $x_i \in \mathbb{Z}_{\geq 0}$ for all $i \in I$