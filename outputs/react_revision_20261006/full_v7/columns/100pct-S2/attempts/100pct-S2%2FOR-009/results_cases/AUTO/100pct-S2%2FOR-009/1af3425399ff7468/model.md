##### Mathematical Model

Let:
- $I$ = set of areas (indexed by $i$), where each area corresponds to a unique ProductName in products.csv.
- $x_i$ = scale of development per day in area $i$ (decision variable, continuous and nonnegative).
- $v_i$ = Value for area $i$ (from products.csv).
- $w_i$ = Weight for area $i$ (from products.csv), representing the resource consumed per unit scale of development.
- $C$ = overall development capacity (from capacity.csv).

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \geq 0 \quad \forall i \in I
\]

##### Data Mapping

- $I$: All ProductName values in file_1_view_0 (products.csv), in original row order.
- $v_i$: Value column in file_1_view_0, matched to $i$ by ProductName.
- $w_i$: Weight column in file_1_view_0, matched to $i$ by ProductName.
- $C$: Capacity column in file_0_view_0 (capacity.csv), single value.
- $x_i$: Decision variable for each $i \in I$.

**Index sets and parameters are not to be sorted or abbreviated; preserve original file and row order.**