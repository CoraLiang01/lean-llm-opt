Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Let $S$ be the set of storage areas (indexed by StorageID):

  $S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15\}$

  with capacities:

  \[
  \begin{aligned}
  &\text{StorageID 1: } 1083 \\
  &\text{StorageID 2: } 1840 \\
  &\text{StorageID 3: } 770 \\
  &\text{StorageID 4: } 1299 \\
  &\text{StorageID 5: } 1259 \\
  &\text{StorageID 6: } 543 \\
  &\text{StorageID 7: } 1831 \\
  &\text{StorageID 8: } 855 \\
  &\text{StorageID 9: } 619 \\
  &\text{StorageID 10: } 637 \\
  &\text{StorageID 11: } 935 \\
  &\text{StorageID 12: } 626 \\
  &\text{StorageID 13: } 1457 \\
  &\text{StorageID 14: } 1198 \\
  &\text{StorageID 15: } 837 \\
  \end{aligned}
  \]

- Let $P$ be the set of air conditioner types (indexed by ProductName):

  \[
  \begin{aligned}
  &\text{Window Unit: Value } 4811, \text{ Weight } 114 \\
  &\text{Portable Unit: Value } 1130, \text{ Weight } 200 \\
  &\text{Split System: Value } 1611, \text{ Weight } 106 \\
  &\text{Ductless System: Value } 3368, \text{ Weight } 256 \\
  &\text{Central AC: Value } 2135, \text{ Weight } 268 \\
  &\text{Hybrid AC: Value } 1046, \text{ Weight } 185 \\
  &\text{Geothermal AC: Value } 4030, \text{ Weight } 299 \\
  &\text{Smart AC: Value } 3761, \text{ Weight } 131 \\
  &\text{Evaporative Cooler: Value } 3523, \text{ Weight } 139 \\
  &\text{Package Unit: Value } 1701, \text{ Weight } 105 \\
  \end{aligned}
  \]

Let $v_j$ be the Value and $w_j$ be the Weight (size) of product $j$.

Let $C_i$ be the Capacity of storage area $i$.

---

### Mathematical Model

**Decision Variables:**

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S, \forall j \in P
\]

**Objective:**

\[
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
\]

**Subject to:**

For each storage area $i \in S$:

\[
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i
\]

For all $i \in S$, $j \in P$:

\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

#### Explicit Data

- Storage areas (StorageID) and their capacities ($C_i$):

  | StorageID | Capacity |
  |-----------|----------|
  | 1         | 1083     |
  | 2         | 1840     |
  | 3         | 770      |
  | 4         | 1299     |
  | 5         | 1259     |
  | 6         | 543      |
  | 7         | 1831     |
  | 8         | 855      |
  | 9         | 619      |
  | 10        | 637      |
  | 11        | 935      |
  | 12        | 626      |
  | 13        | 1457     |
  | 14        | 1198     |
  | 15        | 837      |

- Air conditioner types (ProductName), values ($v_j$), and weights ($w_j$):

  | ProductName         | Value | Weight |
  |---------------------|-------|--------|
  | Window Unit         | 4811  | 114    |
  | Portable Unit       | 1130  | 200    |
  | Split System        | 1611  | 106    |
  | Ductless System     | 3368  | 256    |
  | Central AC          | 2135  | 268    |
  | Hybrid AC           | 1046  | 185    |
  | Geothermal AC       | 4030  | 299    |
  | Smart AC            | 3761  | 131    |
  | Evaporative Cooler  | 3523  | 139    |
  | Package Unit        | 1701  | 105    |

---

**Summary:**

\[
\begin{align*}
\max \quad & \sum_{i \in S} \sum_{j \in P} v_j x_{ij} \\
\text{s.t.} \quad & \sum_{j \in P} w_j x_{ij} \leq C_i, \quad \forall i \in S \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in S, j \in P
\end{align*}
\]

where all sets, parameters, and coefficients are as listed above.