Let $x_{ij}$ be the number of units of air conditioner type $j$ (corresponding to ProductName $j$) to be placed in storage area $i$ (corresponding to StorageID $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Storage areas (from capacity.csv):

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

- Capacities (from capacity.csv):

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

- Air conditioner types (from products.csv):

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

**Decision Variables:**

$x_{ij} \in \mathbb{Z}_{\geq 0}$, for all StorageID $i$ and ProductName $j$.

---

**Objective:**

$$
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}} \text{Value}_j \cdot x_{ij}
$$

where $\text{Value}_j$ is as given above for each ProductName.

---

**Constraints:**

For each storage area $i$ (StorageID):

$$
\sum_{j} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
$$

where $\text{Weight}_j$ is as given above for each ProductName, and $\text{Capacity}_i$ is as given above for each StorageID.

For all $i$ and $j$:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Explicitly:**

For each $i \in \{1,2,\ldots,15\}$:

$$
114\,x_{i,\text{Window Unit}} + 200\,x_{i,\text{Portable Unit}} + 106\,x_{i,\text{Split System}} + 256\,x_{i,\text{Ductless System}} + 268\,x_{i,\text{Central AC}} + 185\,x_{i,\text{Hybrid AC}} + 299\,x_{i,\text{Geothermal AC}} + 131\,x_{i,\text{Smart AC}} + 139\,x_{i,\text{Evaporative Cooler}} + 105\,x_{i,\text{Package Unit}} \leq \text{Capacity}_i
$$

with $\text{Capacity}_i$ as listed above for each StorageID $i$.

All $x_{ij} \in \mathbb{Z}_{\geq 0}$.

---

**Summary of Variables and Parameters:**

- $x_{ij}$: Number of units of ProductName $j$ in StorageID $i$, integer, $\geq 0$
- $\text{Value}_j$: As in products.csv
- $\text{Weight}_j$: As in products.csv
- $\text{Capacity}_i$: As in capacity.csv

**Maximize total value of air conditioners allocated, subject to storage area capacities and integer nonnegativity.**