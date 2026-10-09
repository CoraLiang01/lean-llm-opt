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

  $$
  \begin{align*}
  c_1 &= 1083 \\
  c_2 &= 1840 \\
  c_3 &= 770 \\
  c_4 &= 1299 \\
  c_5 &= 1259 \\
  c_6 &= 543 \\
  c_7 &= 1831 \\
  c_8 &= 855 \\
  c_9 &= 619 \\
  c_{10} &= 637 \\
  c_{11} &= 935 \\
  c_{12} &= 626 \\
  c_{13} &= 1457 \\
  c_{14} &= 1198 \\
  c_{15} &= 837 \\
  \end{align*}
  $$

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

Let $v_j$ be the Value and $w_j$ the Weight for product $j$.

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,15\},\ j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}
$$

**Objective:**

$$
\max \sum_{i=1}^{15} \sum_{j} v_j\, x_{ij}
$$

where $v_j$ is as above for each product.

**Constraints:**

For each storage area $i$ (with capacity $c_i$):

$$
\sum_{j} w_j\, x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,15\}
$$

where $w_j$ is as above for each product.

**Explicitly, with all coefficients:**

Let the products be indexed in the order given above.

For each $i = 1,\ldots,15$:

$$
114\, x_{i,\text{Window Unit}} + 200\, x_{i,\text{Portable Unit}} + 106\, x_{i,\text{Split System}} + 256\, x_{i,\text{Ductless System}} + 268\, x_{i,\text{Central AC}} + 185\, x_{i,\text{Hybrid AC}} + 299\, x_{i,\text{Geothermal AC}} + 131\, x_{i,\text{Smart AC}} + 139\, x_{i,\text{Evaporative Cooler}} + 105\, x_{i,\text{Package Unit}} \leq c_i
$$

with $c_i$ as listed above for each StorageID.

**Variable domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i, j
$$

---

**Summary of Sets and Parameters:**

- Storage areas $i$: StorageID $\in$ {1, 2, ..., 15}
- Products $j$: ProductName $\in$ {Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit}
- $v_j$: Value for product $j$ (see above)
- $w_j$: Weight for product $j$ (see above)
- $c_i$: Capacity for storage area $i$ (see above)

---

**Complete Model:**

$$
\begin{align*}
\max\ & \sum_{i=1}^{15} \Big[ 4811\, x_{i,\text{Window Unit}} + 1130\, x_{i,\text{Portable Unit}} + 1611\, x_{i,\text{Split System}} + 3368\, x_{i,\text{Ductless System}} + 2135\, x_{i,\text{Central AC}} \\
&\qquad + 1046\, x_{i,\text{Hybrid AC}} + 4030\, x_{i,\text{Geothermal AC}} + 3761\, x_{i,\text{Smart AC}} + 3523\, x_{i,\text{Evaporative Cooler}} + 1701\, x_{i,\text{Package Unit}} \Big] \\
\text{s.t.}\quad & 114\, x_{i,\text{Window Unit}} + 200\, x_{i,\text{Portable Unit}} + 106\, x_{i,\text{Split System}} + 256\, x_{i,\text{Ductless System}} + 268\, x_{i,\text{Central AC}} \\
&\qquad + 185\, x_{i,\text{Hybrid AC}} + 299\, x_{i,\text{Geothermal AC}} + 131\, x_{i,\text{Smart AC}} + 139\, x_{i,\text{Evaporative Cooler}} + 105\, x_{i,\text{Package Unit}} \leq c_i \\
&\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad \forall i = 1,\ldots,15 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\end{align*}
$$

where $c_i$ is the capacity for StorageID $i$ as listed above.