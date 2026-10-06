Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Let $S$ be the set of storage areas (StorageID from capacity.csv).
- Let $P$ be the set of air conditioner types (ProductName from products.csv).
- $c_i$ = Capacity of storage area $i$ (from capacity.csv).
- $v_j$ = Value of air conditioner type $j$ (from products.csv).
- $w_j$ = Weight (size) of air conditioner type $j$ (from products.csv).

**Sets and Data:**

Storage Areas (from capacity.csv, in source order):

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

Capacities:

- $c_1 = 1083$
- $c_2 = 1840$
- $c_3 = 770$
- $c_4 = 1299$
- $c_5 = 1259$
- $c_6 = 543$
- $c_7 = 1831$
- $c_8 = 855$
- $c_9 = 619$
- $c_{10} = 637$
- $c_{11} = 935$
- $c_{12} = 626$
- $c_{13} = 1457$
- $c_{14} = 1198$
- $c_{15} = 837$

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

**Mathematical Model:**

Objective:
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
\]

Subject to, for each storage area $i \in S$:
\[
\sum_{j \in P} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in S
\]

Integrality and nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S,\, j \in P
\]

**Where:**

- $S = \{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}$
- $P = \{$Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit$\}$
- $c_i$ as listed above
- $v_j$ and $w_j$ as listed above

**Decision variables:**
- $x_{ij}$ = number of units of air conditioner type $j$ placed in storage area $i$, integer, $\geq 0$.

**All data and identifiers are preserved in source order as required.**