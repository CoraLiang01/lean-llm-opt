Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are nonnegative integers.

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

- Capacity of each storage area (from capacity.csv):

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

| ProductName           |
|----------------------|
| Window Unit          |
| Portable Unit        |
| Split System         |
| Ductless System      |
| Central AC           |
| Hybrid AC            |
| Geothermal AC        |
| Smart AC             |
| Evaporative Cooler   |
| Package Unit         |

- Value and Weight (size) of each air conditioner type (from products.csv):

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

### Mathematical Model

**Decision Variables:**

$$
x_{ij} = \text{number of units of air conditioner type } j \text{ placed in storage area } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
$$

where $i \in \{\text{StorageID 1, ..., 15}\}$ and $j \in \{\text{Window Unit, ..., Package Unit}\}$.

---

**Objective:**

$$
\max \sum_{i \in \{\text{1,...,15}\}} \sum_{j \in \{\text{Window Unit, ..., Package Unit}\}} \text{Value}_j \cdot x_{ij}
$$

where $\text{Value}_j$ is as given in the table above.

---

**Constraints:**

For each storage area $i$ (StorageID):

$$
\sum_{j} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
$$

where $\text{Weight}_j$ and $\text{Capacity}_i$ are as given in the tables above.

For all $i, j$:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Parameter Tables (for reference):**

- Storage Areas and Capacities (in source order):

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

- Air Conditioner Types, Values, and Weights (in source order):

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

**Summary of Model:**

- Maximize total value of air conditioners allocated to all storage areas.
- For each storage area, the total weight of allocated units cannot exceed its capacity.
- All allocation variables are nonnegative integers.