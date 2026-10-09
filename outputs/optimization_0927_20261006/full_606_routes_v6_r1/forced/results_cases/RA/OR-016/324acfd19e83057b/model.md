Let $x_i$ denote the number of units of product $i$ to fulfill (allocate to demand), where $i$ indexes the products as identified by the "Product Name" column.

**Parameters:**

- $r_i$: Revenue per unit for product $i$ (from "Revenue")
- $d_i$: Demand for product $i$ (from "Demand")
- $s_i$: Initial Inventory for product $i$ (from "Initial Inventory")

**Objective:**

$$
\max \sum_{i} r_i x_i
$$

**Subject to:**

For each product $i$ (in the order below):

1. **Demand fulfillment cannot exceed demand:**
   $$
   x_i \leq d_i
   $$
2. **Cannot allocate more than available inventory:**
   $$
   x_i \leq s_i
   $$
3. **Nonnegativity and integrality:**
   $$
   x_i \in \mathbb{Z}_{\geq 0}
   $$

**Data (in source order):**

| Product Name           | Revenue | Demand | Initial Inventory |
|------------------------|---------|--------|------------------|
| Beauty - 25            | 25      | 240    | 1570             |
| Beauty - 30            | 30      | 202    | 1330             |
| Beauty - 300           | 300     | 216    | 1420             |
| Beauty - 50            | 50      | 263    | 1700             |
| Beauty - 500           | 500     | 256    | 1690             |
| Clothing - 25          | 25      | 281    | 1840             |
| Clothing - 30          | 30      | 261    | 1710             |
| Clothing - 300         | 300     | 295    | 1930             |
| Clothing - 50          | 50      | 290    | 1890             |
| Clothing - 500         | 500     | 244    | 1570             |
| Electronics - 25       | 25      | 273    | 1810             |
| Electronics - 30       | 30      | 220    | 1410             |
| Electronics - 300      | 300     | 286    | 1830             |
| Electronics - 50       | 50      | 268    | 1750             |
| Electronics - 500      | 500     | 262    | 1690             |
| Home Goods - 25        | 25      | 255    | 1660             |
| Home Goods - 30        | 30      | 218    | 1417             |
| Home Goods - 50        | 50      | 278    | 1807             |
| Home Goods - 300       | 300     | 195    | 1268             |
| Home Goods - 500       | 500     | 248    | 1612             |
| Sports - 25            | 25      | 269    | 1749             |
| Sports - 30            | 30      | 227    | 1476             |
| Sports - 50            | 50      | 285    | 1853             |
| Sports - 300           | 300     | 208    | 1352             |
| Sports - 500           | 500     | 259    | 1684             |
| Furniture - 25         | 25      | 242    | 1573             |
| Furniture - 30         | 30      | 213    | 1385             |
| Furniture - 50         | 50      | 272    | 1768             |
| Furniture - 300        | 300     | 224    | 1456             |
| Furniture - 500        | 500     | 251    | 1632             |
| Toys - 25              | 25      | 257    | 1671             |
| Toys - 30              | 30      | 235    | 1528             |
| Toys - 50              | 50      | 291    | 1892             |
| Toys - 300             | 300     | 199    | 1294             |
| Toys - 500             | 500     | 264    | 1716             |

**Decision variables:**

For each product $i$ in the above table, $x_i$ is the number of units to allocate, with:

$$
0 \leq x_i \leq \min\{d_i, s_i\}, \quad x_i \in \mathbb{Z}
$$

**Complete Model:**

$$
\begin{align*}
\max \quad & \sum_{i} r_i x_i \\
\text{s.t.} \quad & x_i \leq d_i \quad \forall i \\
                  & x_i \leq s_i \quad \forall i \\
                  & x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\end{align*}
$$

where $i$ indexes the products in the table above, with $r_i$, $d_i$, $s_i$ as given.