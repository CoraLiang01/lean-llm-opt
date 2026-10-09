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
\max \sum_{i \in \{\text{1},\ldots,\text{15}\}} \sum_{j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}} v_j \cdot x_{ij}
\]
where $v_j$ is the Value of product $j$.

Subject to, for each storage area $i$ (StorageID):

\[
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{\text{1},\ldots,\text{15}\}
\]
where $w_j$ is the Weight of product $j$, and $C_i$ is the Capacity of storage area $i$.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

**Numerical Data:**

- For $j$ in order: Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit
- $v_j$ (Value): 4811, 1130, 1611, 3368, 2135, 1046, 4030, 3761, 3523, 1701
- $w_j$ (Weight): 114, 200, 106, 256, 268, 185, 299, 131, 139, 105
- $C_i$ (Capacity): as listed above for StorageID 1–15

**Full Formulation:**

\[
\begin{align*}
\max\ & \sum_{i=1}^{15} \Big[4811\,x_{i,\text{Window Unit}} + 1130\,x_{i,\text{Portable Unit}} + 1611\,x_{i,\text{Split System}} + 3368\,x_{i,\text{Ductless System}} \\
&\qquad + 2135\,x_{i,\text{Central AC}} + 1046\,x_{i,\text{Hybrid AC}} + 4030\,x_{i,\text{Geothermal AC}} + 3761\,x_{i,\text{Smart AC}} \\
&\qquad + 3523\,x_{i,\text{Evaporative Cooler}} + 1701\,x_{i,\text{Package Unit}} \Big] \\
\text{s.t.}\quad & 114\,x_{i,\text{Window Unit}} + 200\,x_{i,\text{Portable Unit}} + 106\,x_{i,\text{Split System}} + 256\,x_{i,\text{Ductless System}} \\
&\qquad + 268\,x_{i,\text{Central AC}} + 185\,x_{i,\text{Hybrid AC}} + 299\,x_{i,\text{Geothermal AC}} + 131\,x_{i,\text{Smart AC}} \\
&\qquad + 139\,x_{i,\text{Evaporative Cooler}} + 105\,x_{i,\text{Package Unit}} \leq C_i \qquad \forall i=1,\ldots,15 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,15;\ j\in\{\text{Window Unit}, \ldots, \text{Package Unit}\}
\end{align*}
\]

Where $C_i$ is the Capacity for StorageID $i$ as listed above.