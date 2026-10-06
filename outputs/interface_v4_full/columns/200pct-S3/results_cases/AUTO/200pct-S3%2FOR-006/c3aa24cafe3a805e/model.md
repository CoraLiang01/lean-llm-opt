## Abstract Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (from products.csv, column ProductName)
- For each $i \in I$:
    - $v_i$ = benefit coefficient of vehicle type $i$ (from products.csv, column Value)
    - $w_i$ = inventory weight (space requirement) of vehicle type $i$ (from products.csv, column Weight)
- $C$ = total inventory capacity (from capacity.csv, column Capacity)
- $x_i$ = number of units of vehicle type $i$ to order daily (decision variable, integer, $x_i \geq 0$)

### Objective
$$
\max \sum_{i \in I} v_i x_i
$$

### Constraints
$$
\sum_{i \in I} w_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

---

## Data Mapping

- $I$: All records in file_1_view_0 (products.csv), column ProductName
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity, source_row 0

All parameters and index sets are mapped directly from the returned CSVQA data, preserving original file and column names.