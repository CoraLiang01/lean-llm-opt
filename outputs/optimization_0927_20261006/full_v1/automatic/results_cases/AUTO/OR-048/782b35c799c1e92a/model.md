Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Storage areas (from capacity.csv, in order):

  1. StorageID = 1, Capacity = 1083  
  2. StorageID = 2, Capacity = 1840  
  3. StorageID = 3, Capacity = 770  
  4. StorageID = 4, Capacity = 1299  
  5. StorageID = 5, Capacity = 1259  
  6. StorageID = 6, Capacity = 543  
  7. StorageID = 7, Capacity = 1831  
  8. StorageID = 8, Capacity = 855  
  9. StorageID = 9, Capacity = 619  
  10. StorageID = 10, Capacity = 637  
  11. StorageID = 11, Capacity = 935  
  12. StorageID = 12, Capacity = 626  
  13. StorageID = 13, Capacity = 1457  
  14. StorageID = 14, Capacity = 1198  
  15. StorageID = 15, Capacity = 837  

- Air conditioner types (from products.csv, in order):

  1. ProductName = Window Unit, Value = 4811, Weight = 114  
  2. ProductName = Portable Unit, Value = 1130, Weight = 200  
  3. ProductName = Split System, Value = 1611, Weight = 106  
  4. ProductName = Ductless System, Value = 3368, Weight = 256  
  5. ProductName = Central AC, Value = 2135, Weight = 268  
  6. ProductName = Hybrid AC, Value = 1046, Weight = 185  
  7. ProductName = Geothermal AC, Value = 4030, Weight = 299  
  8. ProductName = Smart AC, Value = 3761, Weight = 131  
  9. ProductName = Evaporative Cooler, Value = 3523, Weight = 139  
  10. ProductName = Package Unit, Value = 1701, Weight = 105  

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} = \text{number of units of air conditioner type } j \text{ placed in storage area } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
$$

where $i \in \{1,2,\ldots,15\}$ (StorageID), $j \in \{1,2,\ldots,10\}$ (ProductName order above).

---

**Objective:**

$$
\max \sum_{i=1}^{15} \sum_{j=1}^{10} v_j \cdot x_{ij}
$$

where $v_j$ is the Value of product $j$ (see above).

---

**Constraints:**

For each storage area $i$ (StorageID):

$$
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,15
$$

where $w_j$ is the Weight of product $j$, and $C_i$ is the Capacity of storage area $i$.

For all $i, j$:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Parameter Table (in source order):**

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

| ProductName           | Value | Weight |
|-----------------------|-------|--------|
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

---

**Summary:**

- Maximize total value of air conditioners allocated to storage areas.
- For each storage area, total weight of allocated units cannot exceed its capacity.
- All allocations are nonnegative integers.
- All identifiers and coefficients are as retrieved and in original order.