Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Storage areas (from capacity.csv, in source order):

  | StorageID |
  |-----------|
  | 1         |
  | 2         |
  | 3         |
  | 4         |
  | 5         |
  | 6         |
  | 7         |
  | 8         |
  | 9         |
  | 10        |
  | 11        |
  | 12        |
  | 13        |
  | 14        |
  | 15        |

  With capacities:

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

- Air conditioner types (from products.csv, in source order):

  | ProductName           | Value | Weight |
  |---------------------- |-------|--------|
  | Window Unit           | 4811  | 114    |
  | Portable Unit         | 1130  | 200    |
  | Split System          | 1611  | 106    |
  | Ductless System       | 3368  | 256    |
  | Central AC            | 2135  | 268    |
  | Hybrid AC             | 1046  | 185    |
  | Geothermal AC         | 4030  | 299    |
  | Smart AC              | 3761  | 131    |
  | Evaporative Cooler    | 3523  | 139    |
  | Package Unit          | 1701  | 105    |

**Mathematical Model:**

**Decision Variables:**

$$
x_{ij} = \text{number of units of air conditioner type } j \text{ placed in storage area } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
$$

where $i \in \{\text{1, 2, ..., 15}\}$ (StorageID), $j \in \{$Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit$\}$ (ProductName).

---

**Objective:**

$$
\max \sum_{i \in \{\text{1,...,15}\}} \sum_{j \in \{\text{all ProductNames}\}} v_j \cdot x_{ij}
$$

where $v_j$ is the Value of product $j$:

- Window Unit: 4811
- Portable Unit: 1130
- Split System: 1611
- Ductless System: 3368
- Central AC: 2135
- Hybrid AC: 1046
- Geothermal AC: 4030
- Smart AC: 3761
- Evaporative Cooler: 3523
- Package Unit: 1701

---

**Constraints:**

For each storage area $i$ (StorageID):

$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{\text{1,...,15}\}
$$

where $w_j$ is the Weight of product $j$:

- Window Unit: 114
- Portable Unit: 200
- Split System: 106
- Ductless System: 256
- Central AC: 268
- Hybrid AC: 185
- Geothermal AC: 299
- Smart AC: 131
- Evaporative Cooler: 139
- Package Unit: 105

and $C_i$ is the Capacity of storage area $i$ (see table above).

---

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**Complete Model:**

$$
\begin{align*}
\max \quad & \sum_{i=1}^{15} \sum_{j=1}^{10} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \qquad \forall i=1,\ldots,15 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,15;\ j=1,\ldots,10
\end{align*}
$$

where the mapping of $j$ to ProductName, $v_j$, and $w_j$ is as listed above, and $C_i$ is the Capacity for StorageID $i$ as given.