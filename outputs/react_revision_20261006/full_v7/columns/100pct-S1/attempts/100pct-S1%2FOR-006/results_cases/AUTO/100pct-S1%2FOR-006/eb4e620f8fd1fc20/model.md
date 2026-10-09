## Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (from all ProductName in products.csv)
- For each $i \in I$:
    - $v_i$ = Value of vehicle type $i$ (from Value in products.csv)
    - $w_i$ = Weight of vehicle type $i$ (from Weight in products.csv)
- $C$ = total inventory capacity (from Capacity in capacity.csv)
- $x_i$ = number of units of vehicle type $i$ to order daily (decision variable, integer, $x_i \geq 0$)

### Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

### Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

## Data Mapping

- $I$: All records in `file_1_view_0` (products.csv), column `ProductName`
- $v_i$: `file_1_view_0`, column `Value`, keyed by `ProductName`
- $w_i$: `file_1_view_0`, column `Weight`, keyed by `ProductName`
- $C$: `file_0_view_0`, column `Capacity`
- $x_i$: Decision variable for each $i \in I$ (vehicle type from `ProductName`)

All parameters are mapped directly from the specified columns and files. The model maximizes total benefit from daily vehicle orders, subject to the overall inventory capacity, with integer nonnegative order quantities for each vehicle type.