Let:
- $I$ = set of all products, indexed by $i$ (with each product uniquely identified by its "Product Name" from the data).
- For each product $i \in I$:
    - $r_i$ = Revenue per unit (from the "Revenue" column)
    - $d_i$ = Demand (from the "Demand" column)
    - $s_i$ = Initial Inventory (from the "Initial Inventory" column)
- Decision variable: $x_i$ = number of units of product $i$ to fulfill (allocate to demand), $x_i \in \mathbb{Z}_{\geq 0}$

#### Mathematical Model

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Subject to:**
\[
\begin{align*}
& x_i \leq d_i, && \forall i \in I \quad \text{(cannot fulfill more than demand)} \\
& x_i \leq s_i, && \forall i \in I \quad \text{(cannot fulfill more than available inventory)} \\
& x_i \geq 0 \text{ and integer}, && \forall i \in I \\
\end{align*}
\]

**Where:**

- $I$ = 
    - Beauty - 25
    - Beauty - 30
    - Beauty - 300
    - Beauty - 50
    - Beauty - 500
    - Clothing - 25
    - Clothing - 30
    - Clothing - 300
    - Clothing - 50
    - Clothing - 500
    - Electronics - 25
    - Electronics - 30
    - Electronics - 300
    - Electronics - 50
    - Electronics - 500
    - Home Goods - 25
    - Home Goods - 30
    - Home Goods - 50
    - Home Goods - 300
    - Home Goods - 500
    - Sports - 25
    - Sports - 30
    - Sports - 50
    - Sports - 300
    - Sports - 500
    - Furniture - 25
    - Furniture - 30
    - Furniture - 50
    - Furniture - 300
    - Furniture - 500
    - Toys - 25
    - Toys - 30
    - Toys - 50
    - Toys - 300
    - Toys - 500

- For each $i$ (in the same order as above):

    | Product Name           | $r_i$ (Revenue) | $d_i$ (Demand) | $s_i$ (Initial Inventory) |
    |-----------------------|-----------------|---------------|--------------------------|
    | Beauty - 25           | 25              | 240           | 1570                     |
    | Beauty - 30           | 30              | 202           | 1330                     |
    | Beauty - 300          | 300             | 216           | 1420                     |
    | Beauty - 50           | 50              | 263           | 1700                     |
    | Beauty - 500          | 500             | 256           | 1690                     |
    | Clothing - 25         | 25              | 281           | 1840                     |
    | Clothing - 30         | 30              | 261           | 1710                     |
    | Clothing - 300        | 300             | 295           | 1930                     |
    | Clothing - 50         | 50              | 290           | 1890                     |
    | Clothing - 500        | 500             | 244           | 1570                     |
    | Electronics - 25      | 25              | 273           | 1810                     |
    | Electronics - 30      | 30              | 220           | 1410                     |
    | Electronics - 300     | 300             | 286           | 1830                     |
    | Electronics - 50      | 50              | 268           | 1750                     |
    | Electronics - 500     | 500             | 262           | 1690                     |
    | Home Goods - 25       | 25              | 255           | 1660                     |
    | Home Goods - 30       | 30              | 218           | 1417                     |
    | Home Goods - 50       | 50              | 278           | 1807                     |
    | Home Goods - 300      | 300             | 195           | 1268                     |
    | Home Goods - 500      | 500             | 248           | 1612                     |
    | Sports - 25           | 25              | 269           | 1749                     |
    | Sports - 30           | 30              | 227           | 1476                     |
    | Sports - 50           | 50              | 285           | 1853                     |
    | Sports - 300          | 300             | 208           | 1352                     |
    | Sports - 500          | 500             | 259           | 1684                     |
    | Furniture - 25        | 25              | 242           | 1573                     |
    | Furniture - 30        | 30              | 213           | 1385                     |
    | Furniture - 50        | 50              | 272           | 1768                     |
    | Furniture - 300       | 300             | 224           | 1456                     |
    | Furniture - 500       | 500             | 251           | 1632                     |
    | Toys - 25             | 25              | 257           | 1671                     |
    | Toys - 30             | 30              | 235           | 1528                     |
    | Toys - 50             | 50              | 291           | 1892                     |
    | Toys - 300            | 300             | 199           | 1294                     |
    | Toys - 500            | 500             | 264           | 1716                     |

**Variable domains:**
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\]

**Summary:**  
Maximize total revenue by choosing, for each product, how many units to fulfill (up to the minimum of demand and available inventory), with all variables being nonnegative integers.