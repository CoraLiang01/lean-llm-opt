Mathematical Model

Index Sets:
Let $I$ be the set of areas, with each area identified by its ProductName in file_1_view_0.

Parameters:
For each $i \in I$:
 $v_i$ = Value for area $i$ (file_1_view_0, column Value)
 $w_i$ = Weight for area $i$ (file_1_view_0, column Weight)
Let $C$ = overall development capacity (file_0_view_0, column Capacity)

Decision Variables:
For each $i \in I$:
 $x_i$ = integer, daily scale of development in area $i$ ($x_i \in \mathbb{Z}_{\geq 0}$)

Objective:
$\max \sum_{i \in I} v_i x_i$

Subject to:
$\sum_{i \in I} w_i x_i \leq C$

$x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$

Data Mapping

Index Sets:
I: file_1_view_0, column ProductName

Parameters:
$v_i$: file_1_view_0, column Value, keyed by ProductName
$w_i$: file_1_view_0, column Weight, keyed by ProductName
$C$: file_0_view_0, column Capacity

Variables:
$x_i$: integer, for each $i \in I$ (ProductName from file_1_view_0)