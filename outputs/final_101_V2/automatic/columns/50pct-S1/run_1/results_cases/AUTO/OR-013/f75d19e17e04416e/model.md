Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Storage areas $i$ (from capacity.csv):

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

- Air conditioner types $j$ (from products.csv):

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

---

**Mathematical Model**

**Decision Variables:**

$$
x_{ij} = \text{number of units of air conditioner type } j \text{ placed in storage area } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
$$

**Objective:**

$$
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{all ProductNames}\}} v_j \cdot x_{ij}
$$

where $v_j$ is the Value of product $j$.

**Constraints:**

For each storage area $i$ (StorageID):

$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,15\}
$$

where $w_j$ is the Weight of product $j$, and $C_i$ is the Capacity of storage area $i$.

For all $i, j$:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Explicitly, using the data:**

Let $i$ index StorageID $\in \{1,2,\ldots,15\}$, $j$ index ProductName as listed above.

**Objective:**

$$
\max \sum_{i=1}^{15} \Big[
4811\, x_{i,\text{Window Unit}} +
1130\, x_{i,\text{Portable Unit}} +
1611\, x_{i,\text{Split System}} +
3368\, x_{i,\text{Ductless System}} +
2135\, x_{i,\text{Central AC}} +
1046\, x_{i,\text{Hybrid AC}} +
4030\, x_{i,\text{Geothermal AC}} +
3761\, x_{i,\text{Smart AC}} +
3523\, x_{i,\text{Evaporative Cooler}} +
1701\, x_{i,\text{Package Unit}}
\Big]
$$

**For each storage area $i$ (with its Capacity $C_i$):**

For $i=1$ (StorageID 1, Capacity 1083):

$$
114\, x_{1,\text{Window Unit}} +
200\, x_{1,\text{Portable Unit}} +
106\, x_{1,\text{Split System}} +
256\, x_{1,\text{Ductless System}} +
268\, x_{1,\text{Central AC}} +
185\, x_{1,\text{Hybrid AC}} +
299\, x_{1,\text{Geothermal AC}} +
131\, x_{1,\text{Smart AC}} +
139\, x_{1,\text{Evaporative Cooler}} +
105\, x_{1,\text{Package Unit}}
\leq 1083
$$

Repeat this constraint for each $i=2,\ldots,15$ with the corresponding $C_i$.

**Variable domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,15\},\; j \in \{\text{all ProductNames}\}
$$

---

**All data and constraints are included as required.**