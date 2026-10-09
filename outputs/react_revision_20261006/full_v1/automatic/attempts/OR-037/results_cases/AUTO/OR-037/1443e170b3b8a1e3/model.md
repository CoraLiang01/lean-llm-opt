#### Symbolic Mathematical Model

Let:
- $I$ = set of vehicle types (indexed by $i$), corresponding to all ProductName values in products.csv.
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable), $x_i \in \mathbb{Z}_{\geq 0}$
- $v_i$ = profit per unit of vehicle type $i$ (from Value)
- $w_i$ = weight (inventory space consumed) per unit of vehicle type $i$ (from Weight)
- $C$ = overall inventory capacity (from Capacity in capacity.csv)

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraint:**
\[
\sum_{i \in I} w_i x_i \leq C
\]

**Variable domains:**
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

#### Data Mapping

- $I$: All records in file_1_view_0 (products.csv), column ProductName.
- $v_i$: file_1_view_0, column Value, for each $i$.
- $w_i$: file_1_view_0, column Weight, for each $i$.
- $C$: file_0_view_0, column Capacity.
- $x_i$: Decision variable for each $i \in I$.

**Table references:**
- file_0_view_0: capacity.csv, column Capacity (overall inventory capacity)
- file_1_view_0: products.csv, columns ProductName (vehicle type), Value (profit), Weight (inventory space per unit)

---

**Index sets, parameters, and constraints are defined using all rows as returned above, preserving original order and identifiers.**