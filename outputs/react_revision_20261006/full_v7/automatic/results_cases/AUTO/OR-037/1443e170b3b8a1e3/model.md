#### Mathematical Model

Let:
- $I$ = set of vehicle types (indexed by $i$), as given by the ProductName column in products.csv.
- For each $i \in I$:
    - $p_i$ = profit per unit of vehicle $i$ (Value column, products.csv)
    - $w_i$ = inventory space required per unit of vehicle $i$ (Weight column, products.csv)
- $C$ = overall inventory capacity (Capacity column, capacity.csv)
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable)

**Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
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

- $I$: All records in products.csv, column ProductName, table_id: file_1_view_0
- $p_i$: products.csv, column Value, table_id: file_1_view_0, keyed by ProductName
- $w_i$: products.csv, column Weight, table_id: file_1_view_0, keyed by ProductName
- $C$: capacity.csv, column Capacity, table_id: file_0_view_0
- $x_i$: Decision variable for each $i \in I$ (vehicle type)