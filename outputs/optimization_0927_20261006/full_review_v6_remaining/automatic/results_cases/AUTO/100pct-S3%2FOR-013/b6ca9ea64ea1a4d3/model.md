Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName $j$) to be placed in storage area $i$ (StorageID $i$). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Storage areas (from capacity.csv, in order):

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

**Mathematical Model:**

Maximize total value:
$$
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}} v_j \cdot x_{ij}
$$

where $v_j$ is the Value for product $j$.

Subject to, for each storage area $i$ (StorageID):

$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,15\}
$$

where $w_j$ is the Weight for product $j$, and $C_i$ is the Capacity for storage area $i$.

And

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

**Explicitly, with all identifiers and coefficients:**

Let $i$ index StorageID $\in \{1,2,\ldots,15\}$, $j$ index ProductName as listed above.

- $C_i$ (Capacity for StorageID $i$):

  - $C_1 = 1083$
  - $C_2 = 1840$
  - $C_3 = 770$
  - $C_4 = 1299$
  - $C_5 = 1259$
  - $C_6 = 543$
  - $C_7 = 1831$
  - $C_8 = 855$
  - $C_9 = 619$
  - $C_{10} = 637$
  - $C_{11} = 935$
  - $C_{12} = 626$
  - $C_{13} = 1457$
  - $C_{14} = 1198$
  - $C_{15} = 837$

- $v_j$ (Value for ProductName $j$):

  - Window Unit: $v_1 = 4811$
  - Portable Unit: $v_2 = 1130$
  - Split System: $v_3 = 1611$
  - Ductless System: $v_4 = 3368$
  - Central AC: $v_5 = 2135$
  - Hybrid AC: $v_6 = 1046$
  - Geothermal AC: $v_7 = 4030$
  - Smart AC: $v_8 = 3761$
  - Evaporative Cooler: $v_9 = 3523$
  - Package Unit: $v_{10} = 1701$

- $w_j$ (Weight for ProductName $j$):

  - Window Unit: $w_1 = 114$
  - Portable Unit: $w_2 = 200$
  - Split System: $w_3 = 106$
  - Ductless System: $w_4 = 256$
  - Central AC: $w_5 = 268$
  - Hybrid AC: $w_6 = 185$
  - Geothermal AC: $w_7 = 299$
  - Smart AC: $w_8 = 131$
  - Evaporative Cooler: $w_9 = 139$
  - Package Unit: $w_{10} = 105$

**Variables:**

- $x_{ij}$: integer, $\geq 0$, for all $i \in \{1,\ldots,15\}$, $j \in \{1,\ldots,10\}$

**Full Model:**

$$
\max \sum_{i=1}^{15} \left(
4811\,x_{i,1} + 1130\,x_{i,2} + 1611\,x_{i,3} + 3368\,x_{i,4} + 2135\,x_{i,5} + 1046\,x_{i,6} + 4030\,x_{i,7} + 3761\,x_{i,8} + 3523\,x_{i,9} + 1701\,x_{i,10}
\right)
$$

Subject to, for each $i=1,\ldots,15$:

$$
114\,x_{i,1} + 200\,x_{i,2} + 106\,x_{i,3} + 256\,x_{i,4} + 268\,x_{i,5} + 185\,x_{i,6} + 299\,x_{i,7} + 131\,x_{i,8} + 139\,x_{i,9} + 105\,x_{i,10} \leq C_i
$$

with $C_i$ as above for each StorageID.

and

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,15;\ j=1,\ldots,10
$$

where the mapping of $j$ to ProductName is as listed above.