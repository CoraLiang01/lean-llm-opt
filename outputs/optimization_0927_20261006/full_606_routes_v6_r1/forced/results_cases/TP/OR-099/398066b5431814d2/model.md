##### Sets and Indices

- Warehouses $i \in I = \{1,2,3,4,5,6,7,8,9,10,11\}$
- Stores $j \in J = \{1,2,3,4,5,6,7,8,9,10,11\}$

##### Parameters

From PotentialWarehouses_Costs.csv (in source order):

| $i$ | $f_i$ (Opening Cost) | $u_i$ (Capacity) |
|----|----------------------|------------------|
| 1  | 3000                 | 180              |
| 2  | 3200                 | 160              |
| 3  | 3100                 | 200              |
| 4  | 2800                 | 150              |
| 5  | 3500                 | 170              |
| 6  | 2700                 | 190              |
| 7  | 2900                 | 160              |
| 8  | 3050                 | 175              |
| 9  | 3100                 | 170              |
| 10 | 2200                 | 180              |
| 11 | 2890                 | 190              |

From Stores_Demands.csv (in source order):

| $j$ | $d_j$ (Demand) |
|----|----------------|
| 1  | 30             |
| 2  | 40             |
| 3  | 20             |
| 4  | 35             |
| 5  | 20             |
| 6  | 25             |
| 7  | 45             |
| 8  | 38             |
| 9  | 32             |
| 10 | 41             |
| 11 | 44             |

From TransportationCost.csv (rows: stores $j$, columns: warehouses $i$):

Let $c_{ij}$ be the cost to transport one unit from warehouse $i$ to store $j$:

| $j$ | $c_{1j}$ | $c_{2j}$ | $c_{3j}$ | $c_{4j}$ | $c_{5j}$ | $c_{6j}$ | $c_{7j}$ | $c_{8j}$ | $c_{9j}$ | $c_{10j}$ | $c_{11j}$ |
|-----|----------|----------|----------|----------|----------|----------|----------|----------|----------|-----------|-----------|
| 1   | 12       | 11       | 14       | 15       | 17       | 13       | 12       | 16       | 16       | 14        | 15        |
| 2   | 17       | 19       | 15       | 20       | 18       | 14       | 17       | 15       | 13       | 15        | 16        |
| 3   | 13       | 14       | 12       | 14       | 16       | 15       | 11       | 14       | 16       | 18        | 17        |
| 4   | 18       | 16       | 17       | 13       | 18       | 17       | 14       | 19       | 16       | 13        | 18        |
| 5   | 10       | 13       | 12       | 19       | 15       | 11       | 12       | 14       | 12       | 15        | 17        |
| 6   | 15       | 12       | 14       | 16       | 13       | 17       | 16       | 16       | 14       | 18        | 19        |
| 7   | 14       | 13       | 15       | 17       | 12       | 13       | 14       | 15       | 12       | 16        | 14        |
| 8   | 19       | 16       | 18       | 20       | 17       | 19       | 16       | 18       | 15       | 15        | 18        |
| 9   | 17       | 18       | 12       | 14       | 16       | 15       | 14       | 17       | 21       | 15        | 18        |
| 10  | 14       | 13       | 15       | 17       | 16       | 18       | 14       | 19       | 15       | 17        | 19        |
| 11  | 15       | 13       | 16       | 17       | 11       | 13       | 14       | 15       | 19       | 21        | 13        |

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise.
- $x_{ij} \geq 0$: amount shipped from warehouse $i$ to store $j$ (continuous).

##### Objective Function

Minimize total cost (opening + transportation):

$$
\min \sum_{i=1}^{11} f_i y_i + \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** Each store's demand must be met.
   $$
   \sum_{i=1}^{11} x_{ij} \geq d_j \quad \forall j=1,\ldots,11
   $$

2. **Warehouse capacity:** Each warehouse cannot ship more than its capacity, and only if opened.
   $$
   \sum_{j=1}^{11} x_{ij} \leq u_i y_i \quad \forall i=1,\ldots,11
   $$

3. **Non-negativity and binary:**
   $$
   x_{ij} \geq 0 \quad \forall i=1,\ldots,11;\ j=1,\ldots,11
   $$
   $$
   y_i \in \{0,1\} \quad \forall i=1,\ldots,11
   $$

##### Full Numerical Model

Let $f_i$, $u_i$, $d_j$, $c_{ij}$ as above.

$$
\begin{align*}
\min\ & 3000y_1 + 3200y_2 + 3100y_3 + 2800y_4 + 3500y_5 + 2700y_6 + 2900y_7 + 3050y_8 + 3100y_9 + 2200y_{10} + 2890y_{11} \\
&+ \Big[12x_{11} + 11x_{21} + 14x_{31} + 15x_{41} + 17x_{51} + 13x_{61} + 12x_{71} + 16x_{81} + 16x_{91} + 14x_{10,1} + 15x_{11,1} \\
&+ 17x_{12} + 19x_{22} + 15x_{32} + 20x_{42} + 18x_{52} + 14x_{62} + 17x_{72} + 15x_{82} + 13x_{92} + 15x_{10,2} + 16x_{11,2} \\
&+ 13x_{13} + 14x_{23} + 12x_{33} + 14x_{43} + 16x_{53} + 15x_{63} + 11x_{73} + 14x_{83} + 16x_{93} + 18x_{10,3} + 17x_{11,3} \\
&+ 18x_{14} + 16x_{24} + 17x_{34} + 13x_{44} + 18x_{54} + 17x_{64} + 14x_{74} + 19x_{84} + 16x_{94} + 13x_{10,4} + 18x_{11,4} \\
&+ 10x_{15} + 13x_{25} + 12x_{35} + 19x_{45} + 15x_{55} + 11x_{65} + 12x_{75} + 14x_{85} + 12x_{95} + 15x_{10,5} + 17x_{11,5} \\
&+ 15x_{16} + 12x_{26} + 14x_{36} + 16x_{46} + 13x_{56} + 17x_{66} + 16x_{76} + 16x_{86} + 14x_{96} + 18x_{10,6} + 19x_{11,6} \\
&+ 14x_{17} + 13x_{27} + 15x_{37} + 17x_{47} + 12x_{57} + 13x_{67} + 14x_{77} + 15x_{87} + 12x_{97} + 16x_{10,7} + 14x_{11,7} \\
&+ 19x_{18} + 16x_{28} + 18x_{38} + 20x_{48} + 17x_{58} + 19x_{68} + 16x_{78} + 18x_{88} + 15x_{98} + 15x_{10,8} + 18x_{11,8} \\
&+ 17x_{19} + 18x_{29} + 12x_{39} + 14x_{49} + 16x_{59} + 15x_{69} + 14x_{79} + 17x_{89} + 21x_{99} + 15x_{10,9} + 18x_{11,9} \\
&+ 14x_{1,10} + 13x_{2,10} + 15x_{3,10} + 17x_{4,10} + 16x_{5,10} + 18x_{6,10} + 14x_{7,10} + 19x_{8,10} + 15x_{9,10} + 17x_{10,10} + 19x_{11,10} \\
&+ 15x_{1,11} + 13x_{2,11} + 16x_{3,11} + 17x_{4,11} + 11x_{5,11} + 13x_{6,11} + 14x_{7,11} + 15x_{8,11} + 19x_{9,11} + 21x_{10,11} + 13x_{11,11} \Big]
\end{align*}
$$

Subject to:

For each store $j=1,\ldots,11$:
$$
x_{1j} + x_{2j} + x_{3j} + x_{4j} + x_{5j} + x_{6j} + x_{7j} + x_{8j} + x_{9j} + x_{10,j} + x_{11,j} \geq d_j
$$

For each warehouse $i=1,\ldots,11$:
$$
x_{i1} + x_{i2} + x_{i3} + x_{i4} + x_{i5} + x_{i6} + x_{i7} + x_{i8} + x_{i9} + x_{i,10} + x_{i,11} \leq u_i y_i
$$

Variable domains:
$$
x_{ij} \geq 0 \quad \forall i,j
$$
$$
y_i \in \{0,1\} \quad \forall i
$$

##### Retrieved Information

PotentialWarehouses_Costs.csv (source order):

| Warehouse (i) | Opening Cost (fi) | Capacity (units) |
|---------------|-------------------|------------------|
| 1             | 3000              | 180              |
| 2             | 3200              | 160              |
| 3             | 3100              | 200              |
| 4             | 2800              | 150              |
| 5             | 3500              | 170              |
| 6             | 2700              | 190              |
| 7             | 2900              | 160              |
| 8             | 3050              | 175              |
| 9             | 3100              | 170              |
| 10            | 2200              | 180              |
| 11            | 2890              | 190              |

Stores_Demands.csv (source order):

| Store (j) | Demand (units, dj) |
|-----------|--------------------|
| 1         | 30                 |
| 2         | 40                 |
| 3         | 20                 |
| 4         | 35                 |
| 5         | 20                 |
| 6         | 25                 |
| 7         | 45                 |
| 8         | 38                 |
| 9         | 32                 |
| 10        | 41                 |
| 11        | 44                 |

TransportationCost.csv (source order):

Rows: $j=1,\ldots,11$ (stores), Columns: $i=1,\ldots,11$ (warehouses), as shown in the table above.