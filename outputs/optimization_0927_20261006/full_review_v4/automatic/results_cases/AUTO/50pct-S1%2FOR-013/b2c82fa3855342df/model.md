Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Storage areas (from capacity.csv, in source order):

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

Objective:
\[
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{Window Unit},\,\text{Portable Unit},\,\text{Split System},\,\text{Ductless System},\,\text{Central AC},\,\text{Hybrid AC},\,\text{Geothermal AC},\,\text{Smart AC},\,\text{Evaporative Cooler},\,\text{Package Unit}\}} v_j \cdot x_{ij}
\]
where $v_j$ is the Value of product $j$.

Subject to, for each storage area $i$ (StorageID):

\[
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,15\}
\]
where $w_j$ is the Weight of product $j$, and $C_i$ is the Capacity of storage area $i$.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i,\,j
\]

**Parameter values:**

- $C_i$ (Capacity for each StorageID):

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

- $(v_j, w_j)$ for each product $j$ (in source order):

  - Window Unit: $v = 4811$, $w = 114$
  - Portable Unit: $v = 1130$, $w = 200$
  - Split System: $v = 1611$, $w = 106$
  - Ductless System: $v = 3368$, $w = 256$
  - Central AC: $v = 2135$, $w = 268$
  - Hybrid AC: $v = 1046$, $w = 185$
  - Geothermal AC: $v = 4030$, $w = 299$
  - Smart AC: $v = 3761$, $w = 131$
  - Evaporative Cooler: $v = 3523$, $w = 139$
  - Package Unit: $v = 1701$, $w = 105$

**Complete Model:**

\[
\begin{align*}
\max\ & \sum_{i=1}^{15} \Big[ 4811\,x_{i,\text{Window Unit}} + 1130\,x_{i,\text{Portable Unit}} + 1611\,x_{i,\text{Split System}} + 3368\,x_{i,\text{Ductless System}} \\
&\qquad + 2135\,x_{i,\text{Central AC}} + 1046\,x_{i,\text{Hybrid AC}} + 4030\,x_{i,\text{Geothermal AC}} \\
&\qquad + 3761\,x_{i,\text{Smart AC}} + 3523\,x_{i,\text{Evaporative Cooler}} + 1701\,x_{i,\text{Package Unit}} \Big] \\
\text{s.t.}\quad & 114\,x_{1,\text{Window Unit}} + 200\,x_{1,\text{Portable Unit}} + 106\,x_{1,\text{Split System}} + 256\,x_{1,\text{Ductless System}} \\
&\qquad + 268\,x_{1,\text{Central AC}} + 185\,x_{1,\text{Hybrid AC}} + 299\,x_{1,\text{Geothermal AC}} \\
&\qquad + 131\,x_{1,\text{Smart AC}} + 139\,x_{1,\text{Evaporative Cooler}} + 105\,x_{1,\text{Package Unit}} \leq 1083 \\
& \vdots \\
& 114\,x_{15,\text{Window Unit}} + 200\,x_{15,\text{Portable Unit}} + 106\,x_{15,\text{Split System}} + 256\,x_{15,\text{Ductless System}} \\
&\qquad + 268\,x_{15,\text{Central AC}} + 185\,x_{15,\text{Hybrid AC}} + 299\,x_{15,\text{Geothermal AC}} \\
&\qquad + 131\,x_{15,\text{Smart AC}} + 139\,x_{15,\text{Evaporative Cooler}} + 105\,x_{15,\text{Package Unit}} \leq 837 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,15\},\ j \in \{\text{Window Unit},\,\ldots,\,\text{Package Unit}\}
\end{align*}
\]