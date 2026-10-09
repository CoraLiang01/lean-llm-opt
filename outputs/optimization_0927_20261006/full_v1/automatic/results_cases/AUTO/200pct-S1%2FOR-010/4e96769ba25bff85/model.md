Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are integer and nonnegative.

**Indices:**
- $i \in \{1,2,3,4,5,6,7,8\}$ (SectionID from capacity.csv)
- $j \in \{1,2,3,4,5,6,7,8,9,10\}$ (ProductName from products.csv)

**Parameters:**
- $c_i$ = Capacity of section $i$ (from "Capacity" in capacity.csv)
- $p_j$ = Value (price) of product $j$ (from "Value" in products.csv)
- $w_j$ = Weight (shelf space requirement) of product $j$ (from "Weight" in products.csv)

**Data:**

Sections (from capacity.csv, in source order):

| SectionID | Capacity |
|-----------|----------|
| 1         | 100      |
| 2         | 150      |
| 3         | 120      |
| 4         | 130      |
| 5         | 90       |
| 6         | 110      |
| 7         | 160      |
| 8         | 140      |

Products (from products.csv, in source order):

| ProductName | Value | Weight |
|-------------|-------|--------|
| 1           | 10    | 2      |
| 2           | 15    | 3      |
| 3           | 8     | 1      |
| 4           | 12    | 2      |
| 5           | 20    | 4      |
| 6           | 25    | 5      |
| 7           | 5     | 1      |
| 8           | 30    | 6      |
| 9           | 18    | 3      |
| 10          | 22    | 4      |

---

### Mathematical Model

**Objective:**

$$
\max \sum_{i=1}^{8} \sum_{j=1}^{10} p_j \cdot x_{ij}
$$

That is,

$$
\max \Bigg[
\sum_{i=1}^{8} \Big(
10x_{i1} + 15x_{i2} + 8x_{i3} + 12x_{i4} + 20x_{i5} + 25x_{i6} + 5x_{i7} + 30x_{i8} + 18x_{i9} + 22x_{i10}
\Big)
\Bigg]
$$

**Subject to:**

For each section $i$ (SectionID as below):

$$
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
$$

That is, for each section:

- Section 1: $2x_{11} + 3x_{12} + 1x_{13} + 2x_{14} + 4x_{15} + 5x_{16} + 1x_{17} + 6x_{18} + 3x_{19} + 4x_{1,10} \leq 100$
- Section 2: $2x_{21} + 3x_{22} + 1x_{23} + 2x_{24} + 4x_{25} + 5x_{26} + 1x_{27} + 6x_{28} + 3x_{29} + 4x_{2,10} \leq 150$
- Section 3: $2x_{31} + 3x_{32} + 1x_{33} + 2x_{34} + 4x_{35} + 5x_{36} + 1x_{37} + 6x_{38} + 3x_{39} + 4x_{3,10} \leq 120$
- Section 4: $2x_{41} + 3x_{42} + 1x_{43} + 2x_{44} + 4x_{45} + 5x_{46} + 1x_{47} + 6x_{48} + 3x_{49} + 4x_{4,10} \leq 130$
- Section 5: $2x_{51} + 3x_{52} + 1x_{53} + 2x_{54} + 4x_{55} + 5x_{56} + 1x_{57} + 6x_{58} + 3x_{59} + 4x_{5,10} \leq 90$
- Section 6: $2x_{61} + 3x_{62} + 1x_{63} + 2x_{64} + 4x_{65} + 5x_{66} + 1x_{67} + 6x_{68} + 3x_{69} + 4x_{6,10} \leq 110$
- Section 7: $2x_{71} + 3x_{72} + 1x_{73} + 2x_{74} + 4x_{75} + 5x_{76} + 1x_{77} + 6x_{78} + 3x_{79} + 4x_{7,10} \leq 160$
- Section 8: $2x_{81} + 3x_{82} + 1x_{83} + 2x_{84} + 4x_{85} + 5x_{86} + 1x_{87} + 6x_{88} + 3x_{89} + 4x_{8,10} \leq 140$

**Variable domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

---

**All coefficients, identifiers, and constraints are as retrieved and in original order.**