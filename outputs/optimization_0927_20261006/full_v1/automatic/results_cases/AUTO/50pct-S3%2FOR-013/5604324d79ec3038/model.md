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

  With corresponding capacities:

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

---

**Mathematical Model**

**Decision Variables:**

$$
x_{ij} = \text{number of units of air conditioner type } j \text{ placed in storage area } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
$$

where $i \in \{1,2,\ldots,15\}$ (StorageID), $j \in \{$Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit$\}$ (ProductName, in order above).

---

**Objective:**

$$
\max \sum_{i=1}^{15} \sum_{j=1}^{10} v_j \cdot x_{ij}
$$

where $v_j$ is the Value of product $j$:

- $v_1 = 4811$ (Window Unit)
- $v_2 = 1130$ (Portable Unit)
- $v_3 = 1611$ (Split System)
- $v_4 = 3368$ (Ductless System)
- $v_5 = 2135$ (Central AC)
- $v_6 = 1046$ (Hybrid AC)
- $v_7 = 4030$ (Geothermal AC)
- $v_8 = 3761$ (Smart AC)
- $v_9 = 3523$ (Evaporative Cooler)
- $v_{10} = 1701$ (Package Unit)

---

**Constraints:**

For each storage area $i$ (StorageID), the total weight of all air conditioners placed must not exceed its capacity:

For $i = 1$ (StorageID 1, Capacity 1083):

$$
114\,x_{1,1} + 200\,x_{1,2} + 106\,x_{1,3} + 256\,x_{1,4} + 268\,x_{1,5} + 185\,x_{1,6} + 299\,x_{1,7} + 131\,x_{1,8} + 139\,x_{1,9} + 105\,x_{1,10} \leq 1083
$$

For $i = 2$ (StorageID 2, Capacity 1840):

$$
114\,x_{2,1} + 200\,x_{2,2} + 106\,x_{2,3} + 256\,x_{2,4} + 268\,x_{2,5} + 185\,x_{2,6} + 299\,x_{2,7} + 131\,x_{2,8} + 139\,x_{2,9} + 105\,x_{2,10} \leq 1840
$$

For $i = 3$ (StorageID 3, Capacity 770):

$$
114\,x_{3,1} + 200\,x_{3,2} + 106\,x_{3,3} + 256\,x_{3,4} + 268\,x_{3,5} + 185\,x_{3,6} + 299\,x_{3,7} + 131\,x_{3,8} + 139\,x_{3,9} + 105\,x_{3,10} \leq 770
$$

For $i = 4$ (StorageID 4, Capacity 1299):

$$
114\,x_{4,1} + 200\,x_{4,2} + 106\,x_{4,3} + 256\,x_{4,4} + 268\,x_{4,5} + 185\,x_{4,6} + 299\,x_{4,7} + 131\,x_{4,8} + 139\,x_{4,9} + 105\,x_{4,10} \leq 1299
$$

For $i = 5$ (StorageID 5, Capacity 1259):

$$
114\,x_{5,1} + 200\,x_{5,2} + 106\,x_{5,3} + 256\,x_{5,4} + 268\,x_{5,5} + 185\,x_{5,6} + 299\,x_{5,7} + 131\,x_{5,8} + 139\,x_{5,9} + 105\,x_{5,10} \leq 1259
$$

For $i = 6$ (StorageID 6, Capacity 543):

$$
114\,x_{6,1} + 200\,x_{6,2} + 106\,x_{6,3} + 256\,x_{6,4} + 268\,x_{6,5} + 185\,x_{6,6} + 299\,x_{6,7} + 131\,x_{6,8} + 139\,x_{6,9} + 105\,x_{6,10} \leq 543
$$

For $i = 7$ (StorageID 7, Capacity 1831):

$$
114\,x_{7,1} + 200\,x_{7,2} + 106\,x_{7,3} + 256\,x_{7,4} + 268\,x_{7,5} + 185\,x_{7,6} + 299\,x_{7,7} + 131\,x_{7,8} + 139\,x_{7,9} + 105\,x_{7,10} \leq 1831
$$

For $i = 8$ (StorageID 8, Capacity 855):

$$
114\,x_{8,1} + 200\,x_{8,2} + 106\,x_{8,3} + 256\,x_{8,4} + 268\,x_{8,5} + 185\,x_{8,6} + 299\,x_{8,7} + 131\,x_{8,8} + 139\,x_{8,9} + 105\,x_{8,10} \leq 855
$$

For $i = 9$ (StorageID 9, Capacity 619):

$$
114\,x_{9,1} + 200\,x_{9,2} + 106\,x_{9,3} + 256\,x_{9,4} + 268\,x_{9,5} + 185\,x_{9,6} + 299\,x_{9,7} + 131\,x_{9,8} + 139\,x_{9,9} + 105\,x_{9,10} \leq 619
$$

For $i = 10$ (StorageID 10, Capacity 637):

$$
114\,x_{10,1} + 200\,x_{10,2} + 106\,x_{10,3} + 256\,x_{10,4} + 268\,x_{10,5} + 185\,x_{10,6} + 299\,x_{10,7} + 131\,x_{10,8} + 139\,x_{10,9} + 105\,x_{10,10} \leq 637
$$

For $i = 11$ (StorageID 11, Capacity 935):

$$
114\,x_{11,1} + 200\,x_{11,2} + 106\,x_{11,3} + 256\,x_{11,4} + 268\,x_{11,5} + 185\,x_{11,6} + 299\,x_{11,7} + 131\,x_{11,8} + 139\,x_{11,9} + 105\,x_{11,10} \leq 935
$$

For $i = 12$ (StorageID 12, Capacity 626):

$$
114\,x_{12,1} + 200\,x_{12,2} + 106\,x_{12,3} + 256\,x_{12,4} + 268\,x_{12,5} + 185\,x_{12,6} + 299\,x_{12,7} + 131\,x_{12,8} + 139\,x_{12,9} + 105\,x_{12,10} \leq 626
$$

For $i = 13$ (StorageID 13, Capacity 1457):

$$
114\,x_{13,1} + 200\,x_{13,2} + 106\,x_{13,3} + 256\,x_{13,4} + 268\,x_{13,5} + 185\,x_{13,6} + 299\,x_{13,7} + 131\,x_{13,8} + 139\,x_{13,9} + 105\,x_{13,10} \leq 1457
$$

For $i = 14$ (StorageID 14, Capacity 1198):

$$
114\,x_{14,1} + 200\,x_{14,2} + 106\,x_{14,3} + 256\,x_{14,4} + 268\,x_{14,5} + 185\,x_{14,6} + 299\,x_{14,7} + 131\,x_{14,8} + 139\,x_{14,9} + 105\,x_{14,10} \leq 1198
$$

For $i = 15$ (StorageID 15, Capacity 837):

$$
114\,x_{15,1} + 200\,x_{15,2} + 106\,x_{15,3} + 256\,x_{15,4} + 268\,x_{15,5} + 185\,x_{15,6} + 299\,x_{15,7} + 131\,x_{15,8} + 139\,x_{15,9} + 105\,x_{15,10} \leq 837
$$

---

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,15\},\ j \in \{1,\ldots,10\}
$$

---

**Summary of Indices and Parameters:**

- $i$ indexes StorageID in the order: 1, 2, ..., 15.
- $j$ indexes ProductName in the order: Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit.
- $v_j$ is the Value of product $j$.
- $w_j$ is the Weight of product $j$.
- $C_i$ is the Capacity of storage area $i$.

---

**Complete Model:**

$$
\begin{align*}
\max\ & \sum_{i=1}^{15} \sum_{j=1}^{10} v_j x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{10} w_j x_{ij} \leq C_i, \quad \forall i = 1,\ldots,15 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,15,\ j = 1,\ldots,10
\end{align*}
$$

with all coefficients and indices as above, preserving the original file and row order.