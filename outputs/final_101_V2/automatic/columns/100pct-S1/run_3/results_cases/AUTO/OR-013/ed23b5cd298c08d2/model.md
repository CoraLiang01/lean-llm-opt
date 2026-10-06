Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and $\geq 0$.

Parameters (from products.csv):

| $j$ | ProductName           | Value ($v_j$) | Weight ($w_j$) |
|-----|-----------------------|--------------|---------------|
| 1   | Window Unit           | 4811         | 114           |
| 2   | Portable Unit         | 1130         | 200           |
| 3   | Split System          | 1611         | 106           |
| 4   | Ductless System       | 3368         | 256           |
| 5   | Central AC            | 2135         | 268           |
| 6   | Hybrid AC             | 1046         | 185           |
| 7   | Geothermal AC         | 4030         | 299           |
| 8   | Smart AC              | 3761         | 131           |
| 9   | Evaporative Cooler    | 3523         | 139           |
| 10  | Package Unit          | 1701         | 105           |

Parameters (from capacity.csv):

| $i$ | StorageID | Capacity ($C_i$) |
|-----|-----------|------------------|
| 1   | 1         | 1083             |
| 2   | 2         | 1840             |
| 3   | 3         | 770              |
| 4   | 4         | 1299             |
| 5   | 5         | 1259             |
| 6   | 6         | 543              |
| 7   | 7         | 1831             |
| 8   | 8         | 855              |
| 9   | 9         | 619              |
| 10  | 10        | 637              |
| 11  | 11        | 935              |
| 12  | 12        | 626              |
| 13  | 13        | 1457             |
| 14  | 14        | 1198             |
| 15  | 15        | 837              |

Mathematical Model:

Objective:
$$
\max \sum_{i=1}^{15} \sum_{j=1}^{10} v_j \cdot x_{ij}
$$

Subject to, for each storage area $i$ (StorageID):

$$
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,15\}
$$

Variable domains:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,15\},\ j \in \{1,\ldots,10\}
$$

Where:
- $v_j$ is the Value of product $j$ (from products.csv)
- $w_j$ is the Weight of product $j$ (from products.csv)
- $C_i$ is the Capacity of storage area $i$ (from capacity.csv)
- $x_{ij}$ is the integer number of units of product $j$ placed in storage area $i$

All identifiers and coefficients are as retrieved and in original order.