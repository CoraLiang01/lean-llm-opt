Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Storage areas (from capacity.csv, in order):

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

  Each StorageID $i$ has capacity $C_i$:

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

- Air conditioner types (from products.csv, in order):

  | ProductName           | Value | Weight |
  |----------------------|-------|--------|
  | Window Unit          | 4811  | 114    |
  | Portable Unit        | 1130  | 200    |
  | Split System         | 1611  | 106    |
  | Ductless System      | 3368  | 256    |
  | Central AC           | 2135  | 268    |
  | Hybrid AC            | 1046  | 185    |
  | Geothermal AC        | 4030  | 299    |
  | Smart AC             | 3761  | 131    |
  | Evaporative Cooler   | 3523  | 139    |
  | Package Unit         | 1701  | 105    |

Let $V_j$ be the Value and $W_j$ the Weight (size) of product $j$.

---

**Mathematical Model**

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{StorageID}\},\ j \in \{\text{ProductName}\}
$$

**Objective:**

$$
\max \sum_{i \in \{\text{StorageID}\}} \sum_{j \in \{\text{ProductName}\}} V_j \cdot x_{ij}
$$

**Subject to:**

For each storage area $i$ (StorageID):

$$
\sum_{j \in \{\text{ProductName}\}} W_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{\text{StorageID}\}
$$

Where:

- $C_i$ is the Capacity for StorageID $i$ (see table above)
- $W_j$ is the Weight for ProductName $j$ (see table above)
- $V_j$ is the Value for ProductName $j$ (see table above)

**Variable domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**All identifiers and coefficients are as retrieved and in original order.**