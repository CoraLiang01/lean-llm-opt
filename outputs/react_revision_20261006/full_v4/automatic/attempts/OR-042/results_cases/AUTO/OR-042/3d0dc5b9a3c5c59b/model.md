##### Mathematical Model

Let:
- $I$ = set of drug types, indexed by $i$ (from all ProductName in file_1_view_0)
- $x_i$ = number of units of drug type $i$ to order daily (integer, $\geq 0$)
- $v_i$ = benefit coefficient of drug type $i$ (Value from file_1_view_0)
- $w_i$ = weight per unit of drug type $i$ (Weight from file_1_view_0)
- $C$ = overall inventory capacity (Capacity from file_0_view_0)

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

##### Data Mapping

- $I$: All ProductName in file_1_view_0
- $v_i$: file_1_view_0, column Value, key ProductName
- $w_i$: file_1_view_0, column Weight, key ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: decision variable for each $i \in I$