Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName from products.csv) to be placed in storage area $i$ (StorageID from capacity.csv). All $x_{ij}$ are integer and $\geq 0$.

Maximize total value:
$$
\max \sum_{i \in \{\text{all StorageID}\}} \sum_{j \in \{\text{all ProductName}\}} v_j \cdot x_{ij}
$$

where $v_j$ is the Value of product $j$ from products.csv.

Subject to, for each storage area $i$ (StorageID):

$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i
$$

where $w_j$ is the Weight of product $j$ (from products.csv), and $C_i$ is the Capacity of storage area $i$ (from capacity.csv).

And

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

Numerical Formulation

Let

- Storage areas (from capacity.csv, StorageID and Capacity):

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

- Air conditioner types (from products.csv, ProductName, Value, Weight):

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

Define variables:

- $x_{ij}$: number of units of ProductName $j$ in StorageID $i$, integer $\geq 0$.

Objective:

$$
\max \sum_{i \in \{1,\ldots,15\}} \Big(
4811\,x_{i,\text{Window Unit}} +
1130\,x_{i,\text{Portable Unit}} +
1611\,x_{i,\text{Split System}} +
3368\,x_{i,\text{Ductless System}} +
2135\,x_{i,\text{Central AC}} +
1046\,x_{i,\text{Hybrid AC}} +
4030\,x_{i,\text{Geothermal AC}} +
3761\,x_{i,\text{Smart AC}} +
3523\,x_{i,\text{Evaporative Cooler}} +
1701\,x_{i,\text{Package Unit}}
\Big)
$$

Subject to, for each $i$ (StorageID):

$$
114\,x_{i,\text{Window Unit}} +
200\,x_{i,\text{Portable Unit}} +
106\,x_{i,\text{Split System}} +
256\,x_{i,\text{Ductless System}} +
268\,x_{i,\text{Central AC}} +
185\,x_{i,\text{Hybrid AC}} +
299\,x_{i,\text{Geothermal AC}} +
131\,x_{i,\text{Smart AC}} +
139\,x_{i,\text{Evaporative Cooler}} +
105\,x_{i,\text{Package Unit}}
\leq \text{Capacity}_i
$$

for each $i$ in $\{1,2,\ldots,15\}$, with Capacity$_i$ as above.

And

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,15\},\ j \in \{\text{all 10 ProductName}\}
$$