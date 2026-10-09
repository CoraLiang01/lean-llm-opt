Let $x_{ij}$ be the number of units of product $j$ (with item_name $j$) to be placed on shelf $i$ (with resource_id $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- For each shelf $i$ (resource_id), the capacity is $C_i$ (resource_capacity).
- For each product $j$ (item_name), the value per unit is $v_j$ (item_value), and the weight per unit is $w_j$ (resource_requirement).

**Data:**

- Shelves (from capacity.csv, in source order):

| resource_id | resource_capacity |
|-------------|------------------|
| 1           | 500              |
| 2           | 700              |
| 3           | 600              |
| 4           | 800              |
| 5           | 550              |
| 6           | 900              |
| 7           | 650              |
| 8           | 750              |
| 9           | 820              |
| 10          | 570              |

- Products (from products.csv, in source order):

| item_name | item_value | resource_requirement |
|-----------|------------|---------------------|
| 1         | 50         | 10                  |
| 2         | 70         | 20                  |
| 3         | 30         | 5                   |
| 4         | 60         | 15                  |
| 5         | 80         | 25                  |
| 6         | 90         | 30                  |
| 7         | 40         | 12                  |
| 8         | 100        | 35                  |
| 9         | 55         | 10                  |
| 10        | 75         | 20                  |
| 11        | 65         | 18                  |
| 12        | 95         | 28                  |
| 13        | 45         | 8                   |
| 14        | 85         | 22                  |
| 15        | 70         | 25                  |
| 16        | 110        | 40                  |
| 17        | 50         | 14                  |
| 18        | 60         | 16                  |
| 19        | 120        | 50                  |
| 20        | 100        | 30                  |

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j\, x_{ij}
$$

where $v_j$ is the item_value for product $j$.

**Constraints:**

For each shelf $i$ (resource_id):

$$
\sum_{j=1}^{20} w_j\, x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
$$

where $w_j$ is the resource_requirement for product $j$, and $C_i$ is the resource_capacity for shelf $i$.

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i,j
$$

---

**Explicitly, with all identifiers and coefficients:**

Let $x_{ij}$ = number of units of product with item_name $j$ on shelf with resource_id $i$.

**Objective:**

$$
\max \Bigg[
\sum_{i=1}^{10} \Big(
50\,x_{i1} + 70\,x_{i2} + 30\,x_{i3} + 60\,x_{i4} + 80\,x_{i5} + 90\,x_{i6} + 40\,x_{i7} + 100\,x_{i8} + 55\,x_{i9} + 75\,x_{i10} + 65\,x_{i11} + 95\,x_{i12} + 45\,x_{i13} + 85\,x_{i14} + 70\,x_{i15} + 110\,x_{i16} + 50\,x_{i17} + 60\,x_{i18} + 120\,x_{i19} + 100\,x_{i20}
\Big)
\Bigg]
$$

**Constraints:**

For each shelf $i$:

- For $i=1$ (resource_id 1, resource_capacity 500):

  $$
  10\,x_{1,1} + 20\,x_{1,2} + 5\,x_{1,3} + 15\,x_{1,4} + 25\,x_{1,5} + 30\,x_{1,6} + 12\,x_{1,7} + 35\,x_{1,8} + 10\,x_{1,9} + 20\,x_{1,10} + 18\,x_{1,11} + 28\,x_{1,12} + 8\,x_{1,13} + 22\,x_{1,14} + 25\,x_{1,15} + 40\,x_{1,16} + 14\,x_{1,17} + 16\,x_{1,18} + 50\,x_{1,19} + 30\,x_{1,20} \leq 500
  $$

- For $i=2$ (resource_id 2, resource_capacity 700):

  $$
  10\,x_{2,1} + 20\,x_{2,2} + 5\,x_{2,3} + 15\,x_{2,4} + 25\,x_{2,5} + 30\,x_{2,6} + 12\,x_{2,7} + 35\,x_{2,8} + 10\,x_{2,9} + 20\,x_{2,10} + 18\,x_{2,11} + 28\,x_{2,12} + 8\,x_{2,13} + 22\,x_{2,14} + 25\,x_{2,15} + 40\,x_{2,16} + 14\,x_{2,17} + 16\,x_{2,18} + 50\,x_{2,19} + 30\,x_{2,20} \leq 700
  $$

- For $i=3$ (resource_id 3, resource_capacity 600):

  $$
  10\,x_{3,1} + 20\,x_{3,2} + 5\,x_{3,3} + 15\,x_{3,4} + 25\,x_{3,5} + 30\,x_{3,6} + 12\,x_{3,7} + 35\,x_{3,8} + 10\,x_{3,9} + 20\,x_{3,10} + 18\,x_{3,11} + 28\,x_{3,12} + 8\,x_{3,13} + 22\,x_{3,14} + 25\,x_{3,15} + 40\,x_{3,16} + 14\,x_{3,17} + 16\,x_{3,18} + 50\,x_{3,19} + 30\,x_{3,20} \leq 600
  $$

- For $i=4$ (resource_id 4, resource_capacity 800):

  $$
  10\,x_{4,1} + 20\,x_{4,2} + 5\,x_{4,3} + 15\,x_{4,4} + 25\,x_{4,5} + 30\,x_{4,6} + 12\,x_{4,7} + 35\,x_{4,8} + 10\,x_{4,9} + 20\,x_{4,10} + 18\,x_{4,11} + 28\,x_{4,12} + 8\,x_{4,13} + 22\,x_{4,14} + 25\,x_{4,15} + 40\,x_{4,16} + 14\,x_{4,17} + 16\,x_{4,18} + 50\,x_{4,19} + 30\,x_{4,20} \leq 800
  $$

- For $i=5$ (resource_id 5, resource_capacity 550):

  $$
  10\,x_{5,1} + 20\,x_{5,2} + 5\,x_{5,3} + 15\,x_{5,4} + 25\,x_{5,5} + 30\,x_{5,6} + 12\,x_{5,7} + 35\,x_{5,8} + 10\,x_{5,9} + 20\,x_{5,10} + 18\,x_{5,11} + 28\,x_{5,12} + 8\,x_{5,13} + 22\,x_{5,14} + 25\,x_{5,15} + 40\,x_{5,16} + 14\,x_{5,17} + 16\,x_{5,18} + 50\,x_{5,19} + 30\,x_{5,20} \leq 550
  $$

- For $i=6$ (resource_id 6, resource_capacity 900):

  $$
  10\,x_{6,1} + 20\,x_{6,2} + 5\,x_{6,3} + 15\,x_{6,4} + 25\,x_{6,5} + 30\,x_{6,6} + 12\,x_{6,7} + 35\,x_{6,8} + 10\,x_{6,9} + 20\,x_{6,10} + 18\,x_{6,11} + 28\,x_{6,12} + 8\,x_{6,13} + 22\,x_{6,14} + 25\,x_{6,15} + 40\,x_{6,16} + 14\,x_{6,17} + 16\,x_{6,18} + 50\,x_{6,19} + 30\,x_{6,20} \leq 900
  $$

- For $i=7$ (resource_id 7, resource_capacity 650):

  $$
  10\,x_{7,1} + 20\,x_{7,2} + 5\,x_{7,3} + 15\,x_{7,4} + 25\,x_{7,5} + 30\,x_{7,6} + 12\,x_{7,7} + 35\,x_{7,8} + 10\,x_{7,9} + 20\,x_{7,10} + 18\,x_{7,11} + 28\,x_{7,12} + 8\,x_{7,13} + 22\,x_{7,14} + 25\,x_{7,15} + 40\,x_{7,16} + 14\,x_{7,17} + 16\,x_{7,18} + 50\,x_{7,19} + 30\,x_{7,20} \leq 650
  $$

- For $i=8$ (resource_id 8, resource_capacity 750):

  $$
  10\,x_{8,1} + 20\,x_{8,2} + 5\,x_{8,3} + 15\,x_{8,4} + 25\,x_{8,5} + 30\,x_{8,6} + 12\,x_{8,7} + 35\,x_{8,8} + 10\,x_{8,9} + 20\,x_{8,10} + 18\,x_{8,11} + 28\,x_{8,12} + 8\,x_{8,13} + 22\,x_{8,14} + 25\,x_{8,15} + 40\,x_{8,16} + 14\,x_{8,17} + 16\,x_{8,18} + 50\,x_{8,19} + 30\,x_{8,20} \leq 750
  $$

- For $i=9$ (resource_id 9, resource_capacity 820):

  $$
  10\,x_{9,1} + 20\,x_{9,2} + 5\,x_{9,3} + 15\,x_{9,4} + 25\,x_{9,5} + 30\,x_{9,6} + 12\,x_{9,7} + 35\,x_{9,8} + 10\,x_{9,9} + 20\,x_{9,10} + 18\,x_{9,11} + 28\,x_{9,12} + 8\,x_{9,13} + 22\,x_{9,14} + 25\,x_{9,15} + 40\,x_{9,16} + 14\,x_{9,17} + 16\,x_{9,18} + 50\,x_{9,19} + 30\,x_{9,20} \leq 820
  $$

- For $i=10$ (resource_id 10, resource_capacity 570):

  $$
  10\,x_{10,1} + 20\,x_{10,2} + 5\,x_{10,3} + 15\,x_{10,4} + 25\,x_{10,5} + 30\,x_{10,6} + 12\,x_{10,7} + 35\,x_{10,8} + 10\,x_{10,9} + 20\,x_{10,10} + 18\,x_{10,11} + 28\,x_{10,12} + 8\,x_{10,13} + 22\,x_{10,14} + 25\,x_{10,15} + 40\,x_{10,16} + 14\,x_{10,17} + 16\,x_{10,18} + 50\,x_{10,19} + 30\,x_{10,20} \leq 570
  $$

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

---

**All coefficients, identifiers, and constraints are as retrieved and in original order.**