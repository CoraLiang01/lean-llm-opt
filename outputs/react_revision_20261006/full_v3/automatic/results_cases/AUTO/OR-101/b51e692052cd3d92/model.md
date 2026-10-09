##### Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (from all Product in file_2_view_0 and columns of file_0_view_0 except "Device")
- $K$ = set of devices, indexed by $k$ (from all Device in file_1_view_0 and file_0_view_0)

Parameters:
- $p_i$ = unit profit of product $i$ (from Unit_Profit in file_2_view_0)
- $a_{ki}$ = processing time required by product $i$ on device $k$ (from file_0_view_0, Device $k$, column $i$)
- $c_k$ = monthly capacity of device $k$ (from Monthly_Capacity in file_1_view_0)

Decision variables:
- $x_i \geq 0$ = production quantity of product $i$ (continuous)

Objective:
\[
\max \sum_{i \in I} p_i x_i
\]

Subject to:
\[
\sum_{i \in I} a_{ki} x_i \leq c_k \qquad \forall k \in K
\]
\[
x_i \geq 0 \qquad \forall i \in I
\]

##### Data Mapping

- $I$: All Product in file_2_view_0["Product"] and all columns except "Device" in file_0_view_0
- $K$: All Device in file_1_view_0["Device"] and file_0_view_0["Device"]
- $p_i$: file_2_view_0["Unit_Profit"], key: Product
- $a_{ki}$: file_0_view_0, row: Device $k$, column: $i$
- $c_k$: file_1_view_0["Monthly_Capacity"], key: Device

- Decision variable $x_i$: continuous, nonnegative, for each $i \in I$