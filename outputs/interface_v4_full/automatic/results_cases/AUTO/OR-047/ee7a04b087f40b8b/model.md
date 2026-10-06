#### Abstract Mathematical Model

Let:
- $I$ = set of platforms, indexed by $i$ (from file_0_view_0, column PlatformId)
- $J$ = set of game genres, indexed by $j$ (from file_1_view_0, column ProductName)

Parameters:
- $c_i$ = memory capacity of platform $i$ (file_0_view_0, Capacity, key: PlatformId)
- $v_j$ = value per unit of genre $j$ (file_1_view_0, Value, key: ProductName)
- $w_j$ = memory requirement per unit of genre $j$ (file_1_view_0, Weight, key: ProductName)

Decision variables:
- $x_{ij}$ = number of units of games from genre $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

Subject to:
\[
\sum_{j \in J} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- $I$ (platforms): file_0_view_0, column PlatformId
- $c_i$: file_0_view_0, column Capacity, key: PlatformId
- $J$ (genres): file_1_view_0, column ProductName
- $v_j$: file_1_view_0, column Value, key: ProductName
- $w_j$: file_1_view_0, column Weight, key: ProductName

All variables, parameters, and constraints are indexed and mapped directly to the original file columns and business identifiers. No data or dimension is omitted.