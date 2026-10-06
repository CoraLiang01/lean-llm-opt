Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and $\geq 0$.

Indices:
- $i \in \{1,2,\ldots,15\}$ (StorageID from capacity.csv)
- $j \in \{$Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit$\}$ (ProductName from products.csv)

Parameters:
- $c_i$: Capacity of storage area $i$ (from Capacity column in capacity.csv)
- $v_j$: Value of air conditioner type $j$ (from Value column in products.csv)
- $w_j$: Weight (size) of air conditioner type $j$ (from Weight column in products.csv)

Data:

From capacity.csv (in source order):

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

From products.csv (in source order):

| ProductName         | Value | Weight |
|---------------------|-------|--------|
| Window Unit         | 4811  | 114    |
| Portable Unit       | 1130  | 200    |
| Split System        | 1611  | 106    |
| Ductless System     | 3368  | 256    |
| Central AC          | 2135  | 268    |
| Hybrid AC           | 1046  | 185    |
| Geothermal AC       | 4030  | 299    |
| Smart AC            | 3761  | 131    |
| Evaporative Cooler  | 3523  | 139    |
| Package Unit        | 1701  | 105    |

Model:

Objective:
\[
\max \sum_{i=1}^{15} \sum_{j=1}^{10} v_j x_{ij}
\]
where $v_j$ is as above for each ProductName.

Subject to:

For each storage area $i$ (StorageID from 1 to 15):
\[
\sum_{j=1}^{10} w_j x_{ij} \leq c_i
\]
where $w_j$ is as above for each ProductName, and $c_i$ is as above for each StorageID.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,15;\ j=1,\ldots,10
\]

Explicitly, for each $i$:

- For $i=1$: $\sum_{j=1}^{10} w_j x_{1j} \leq 1083$
- For $i=2$: $\sum_{j=1}^{10} w_j x_{2j} \leq 1840$
- For $i=3$: $\sum_{j=1}^{10} w_j x_{3j} \leq 770$
- For $i=4$: $\sum_{j=1}^{10} w_j x_{4j} \leq 1299$
- For $i=5$: $\sum_{j=1}^{10} w_j x_{5j} \leq 1259$
- For $i=6$: $\sum_{j=1}^{10} w_j x_{6j} \leq 543$
- For $i=7$: $\sum_{j=1}^{10} w_j x_{7j} \leq 1831$
- For $i=8$: $\sum_{j=1}^{10} w_j x_{8j} \leq 855$
- For $i=9$: $\sum_{j=1}^{10} w_j x_{9j} \leq 619$
- For $i=10$: $\sum_{j=1}^{10} w_j x_{10j} \leq 637$
- For $i=11$: $\sum_{j=1}^{10} w_j x_{11j} \leq 935$
- For $i=12$: $\sum_{j=1}^{10} w_j x_{12j} \leq 626$
- For $i=13$: $\sum_{j=1}^{10} w_j x_{13j} \leq 1457$
- For $i=14$: $\sum_{j=1}^{10} w_j x_{14j} \leq 1198$
- For $i=15$: $\sum_{j=1}^{10} w_j x_{15j} \leq 837$

Where for each $j$ (in source order):

- $v_1 = 4811$, $w_1 = 114$ (Window Unit)
- $v_2 = 1130$, $w_2 = 200$ (Portable Unit)
- $v_3 = 1611$, $w_3 = 106$ (Split System)
- $v_4 = 3368$, $w_4 = 256$ (Ductless System)
- $v_5 = 2135$, $w_5 = 268$ (Central AC)
- $v_6 = 1046$, $w_6 = 185$ (Hybrid AC)
- $v_7 = 4030$, $w_7 = 299$ (Geothermal AC)
- $v_8 = 3761$, $w_8 = 131$ (Smart AC)
- $v_9 = 3523$, $w_9 = 139$ (Evaporative Cooler)
- $v_{10} = 1701$, $w_{10} = 105$ (Package Unit)

All $x_{ij} \in \mathbb{Z}_{\geq 0}$.

This is the complete mathematical model as required.