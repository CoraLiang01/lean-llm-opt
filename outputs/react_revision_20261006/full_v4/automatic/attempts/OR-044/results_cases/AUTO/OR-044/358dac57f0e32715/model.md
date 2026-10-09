##### Mathematical Model

Let:
- $I$ = set of SectionIDs from file_0_view_0 (capacity.csv)
- $J$ = set of ProductNames from file_1_view_0 (products.csv)

Parameters:
- $c_i$ = Capacity of section $i$ (from file_0_view_0, column Capacity)
- $v_j$ = Value (price) of product $j$ (from file_1_view_0, column Value)
- $w_j$ = Weight (space requirement) of product $j$ (from file_1_view_0, column Weight)

Decision variables:
- $x_{ij}$ = number of units of product $j$ to be placed in section $i$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

Subject to:
\[
\sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

##### Data Mapping

- $I$: SectionID from file_0_view_0 (capacity.csv)
- $J$: ProductName from file_1_view_0 (products.csv)
- $c_i$: file_0_view_0, column Capacity, key SectionID
- $v_j$: file_1_view_0, column Value, key ProductName
- $w_j$: file_1_view_0, column Weight, key ProductName
- $x_{ij}$: number of units of product $j$ in section $i$ (decision variable, indexed by SectionID and ProductName)