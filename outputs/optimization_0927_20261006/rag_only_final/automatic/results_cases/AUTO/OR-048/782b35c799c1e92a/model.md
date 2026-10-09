Let:
- I = {1, 2, ..., 15} be the set of storage areas, indexed by i, with capacities as given in capacity.csv.
- J = {Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit}, indexed by j, with values and weights as given in products.csv.

Decision variables:
x_{ij} = number of units of air conditioner type j placed in storage area i, for all i in I, j in J.
x_{ij} ∈ {0, 1, 2, ...} (nonnegative integers)

Parameters:
From capacity.csv:
- Capacity_i: capacity of storage area i

| StorageID (i) | Capacity_i |
|---------------|------------|
| 1             | 1083       |
| 2             | 1840       |
| 3             | 770        |
| 4             | 1299       |
| 5             | 1259       |
| 6             | 543        |
| 7             | 1831       |
| 8             | 855        |
| 9             | 619        |
| 10            | 637        |
| 11            | 935        |
| 12            | 626        |
| 13            | 1457       |
| 14            | 1198       |
| 15            | 837        |

From products.csv:
- v_j: value of air conditioner type j
- w_j: weight (size) of air conditioner type j

| ProductName (j)        | v_j  | w_j |
|------------------------|------|-----|
| Window Unit            | 4811 | 114 |
| Portable Unit          | 1130 | 200 |
| Split System           | 1611 | 106 |
| Ductless System        | 3368 | 256 |
| Central AC             | 2135 | 268 |
| Hybrid AC              | 1046 | 185 |
| Geothermal AC          | 4030 | 299 |
| Smart AC               | 3761 | 131 |
| Evaporative Cooler     | 3523 | 139 |
| Package Unit           | 1701 | 105 |

Mathematical Model:

Maximize total value:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{15} \sum_{j \in J} v_j \cdot x_{ij}
\]

Subject to storage area capacity constraints:
For each storage area i ∈ {1,...,15}:
\[
\sum_{j \in J} w_j \cdot x_{ij} \leq \text{Capacity}_i
\]
That is, for each i:

- For i=1: 114 x_{1,Window Unit} + 200 x_{1,Portable Unit} + 106 x_{1,Split System} + 256 x_{1,Ductless System} + 268 x_{1,Central AC} + 185 x_{1,Hybrid AC} + 299 x_{1,Geothermal AC} + 131 x_{1,Smart AC} + 139 x_{1,Evaporative Cooler} + 105 x_{1,Package Unit} ≤ 1083
- For i=2: ... ≤ 1840
- ...
- For i=15: ... ≤ 837

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,...,15\},\ j \in J
\]

Summary:
- Decision variables: x_{ij} = number of units of air conditioner type j in storage area i (nonnegative integers)
- Objective: maximize total value of all air conditioners placed
- Constraints: for each storage area, total size of units placed does not exceed its capacity

All coefficients and identifiers are as given in the supplied CSVs.