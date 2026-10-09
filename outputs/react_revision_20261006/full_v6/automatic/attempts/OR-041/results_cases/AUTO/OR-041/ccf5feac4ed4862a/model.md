##### Mathematical Model

Let:
- $I$ = set of areas (indexed by $i$), with area names as in the ProductName column of products.csv.
- $x_i$ = scale of development per day in area $i$ (decision variable, nonnegative integer).
- $v_i$ = development benefit per unit in area $i$ (from Value column).
- $w_i$ = resource requirement per unit in area $i$ (from Weight column).
- $C$ = overall development capacity (from Capacity column in capacity.csv).

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

##### Data Mapping

- $I$: All area names from file_1_view_0.ProductName (products.csv, ProductName)
- $v_i$: file_1_view_0.Value (products.csv, Value), mapped by ProductName
- $w_i$: file_1_view_0.Weight (products.csv, Weight), mapped by ProductName
- $C$: file_0_view_0.Capacity (capacity.csv, Capacity)
- $x_i$: Decision variable for each $i \in I$ (area)

All parameters are mapped directly from the specified columns and files.