Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- $S$ = set of storage areas (indexed by $i$), with StorageID as below.
- $P$ = set of air conditioner types (indexed by $j$), with ProductName as below.
- $v_j$ = Value of air conditioner type $j$.
- $w_j$ = Weight (size) of air conditioner type $j$.
- $C_i$ = Capacity of storage area $i$.

**Data:**

Storage Areas (from capacity.csv, in source order):

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

Air Conditioner Types (from products.csv, in source order):

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

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S,\, j \in P
$$

**Objective:**

$$
\max \sum_{i \in S} \sum_{j \in P} v_j \, x_{ij}
$$

**Subject to:**

For each storage area $i$ (StorageID):

$$
\sum_{j \in P} w_j \, x_{ij} \leq C_i \qquad \forall i \in S
$$

**Where:**

- $S = \{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}$
- $P = \{$Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit$\}$
- $v_j$ and $w_j$ as in the table above
- $C_i$ as in the table above

**Variable domains:**

$$
x_{ij} \in \{0,1,2,\ldots\} \qquad \forall i \in S,\, j \in P
$$

---

**Complete Numerical Formulation:**

$$
\begin{align*}
\max\quad & \sum_{i=1}^{15} \Big[4811\,x_{i,\text{Window Unit}} + 1130\,x_{i,\text{Portable Unit}} + 1611\,x_{i,\text{Split System}} + 3368\,x_{i,\text{Ductless System}} \\
&\qquad + 2135\,x_{i,\text{Central AC}} + 1046\,x_{i,\text{Hybrid AC}} + 4030\,x_{i,\text{Geothermal AC}} + 3761\,x_{i,\text{Smart AC}} \\
&\qquad + 3523\,x_{i,\text{Evaporative Cooler}} + 1701\,x_{i,\text{Package Unit}} \Big] \\
\text{s.t.}\quad & 114\,x_{1,\text{Window Unit}} + 200\,x_{1,\text{Portable Unit}} + 106\,x_{1,\text{Split System}} + 256\,x_{1,\text{Ductless System}} \\
&\qquad + 268\,x_{1,\text{Central AC}} + 185\,x_{1,\text{Hybrid AC}} + 299\,x_{1,\text{Geothermal AC}} + 131\,x_{1,\text{Smart AC}} \\
&\qquad + 139\,x_{1,\text{Evaporative Cooler}} + 105\,x_{1,\text{Package Unit}} \leq 1083 \\
& \vdots \\
& 114\,x_{15,\text{Window Unit}} + 200\,x_{15,\text{Portable Unit}} + 106\,x_{15,\text{Split System}} + 256\,x_{15,\text{Ductless System}} \\
&\qquad + 268\,x_{15,\text{Central AC}} + 185\,x_{15,\text{Hybrid AC}} + 299\,x_{15,\text{Geothermal AC}} + 131\,x_{15,\text{Smart AC}} \\
&\qquad + 139\,x_{15,\text{Evaporative Cooler}} + 105\,x_{15,\text{Package Unit}} \leq 837 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S,\, j \in P
\end{align*}
$$

All coefficients and identifiers are as retrieved and in original order.