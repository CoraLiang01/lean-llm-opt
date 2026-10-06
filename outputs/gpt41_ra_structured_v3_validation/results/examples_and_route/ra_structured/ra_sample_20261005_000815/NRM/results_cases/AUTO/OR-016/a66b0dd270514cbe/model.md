Let:
- Let $I$ be the set of all products, indexed by $i$, as given in the "Product Name" column.
- For each product $i \in I$:
    - Let $r_i$ = Revenue for product $i$
    - Let $d_i$ = Demand for product $i$
    - Let $s_i$ = Initial Inventory for product $i$
    - Let $x_i$ = quantity of product $i$ to fulfill (decision variable)

The mathematical model is:

---

**Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$ for all $i \in I$

---

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

---

**Constraints:**

1. **Inventory constraint (cannot allocate more than available inventory):**
   \[
   x_i \leq s_i \qquad \forall i \in I
   \]

2. **Demand constraint (cannot fulfill more than demand):**
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]

3. **Nonnegativity and integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
   \]

---

**Where:**

| Product Name              | $r_i$ (Revenue) | $d_i$ (Demand) | $s_i$ (Initial Inventory) |
|--------------------------|-----------------|---------------|--------------------------|
| Beauty - 25              | 25              | 240           | 1570                     |
| Beauty - 30              | 30              | 202           | 1330                     |
| Beauty - 300             | 300             | 216           | 1420                     |
| Beauty - 50              | 50              | 263           | 1700                     |
| Beauty - 500             | 500             | 256           | 1690                     |
| Clothing - 25            | 25              | 281           | 1840                     |
| Clothing - 30            | 30              | 261           | 1710                     |
| Clothing - 300           | 300             | 295           | 1930                     |
| Clothing - 50            | 50              | 290           | 1890                     |
| Clothing - 500           | 500             | 244           | 1570                     |
| Electronics - 25         | 25              | 273           | 1810                     |
| Electronics - 30         | 30              | 220           | 1410                     |
| Electronics - 300        | 300             | 286           | 1830                     |
| Electronics - 50         | 50              | 268           | 1750                     |
| Electronics - 500        | 500             | 262           | 1690                     |
| Home Goods - 25          | 25              | 255           | 1660                     |
| Home Goods - 30          | 30              | 218           | 1417                     |
| Home Goods - 50          | 50              | 278           | 1807                     |
| Home Goods - 300         | 300             | 195           | 1268                     |
| Home Goods - 500         | 500             | 248           | 1612                     |
| Sports - 25              | 25              | 269           | 1749                     |
| Sports - 30              | 30              | 227           | 1476                     |
| Sports - 50              | 50              | 285           | 1853                     |
| Sports - 300             | 300             | 208           | 1352                     |
| Sports - 500             | 500             | 259           | 1684                     |
| Furniture - 25           | 25              | 242           | 1573                     |
| Furniture - 30           | 30              | 213           | 1385                     |
| Furniture - 50           | 50              | 272           | 1768                     |
| Furniture - 300          | 300             | 224           | 1456                     |
| Furniture - 500          | 500             | 251           | 1632                     |
| Toys - 25                | 25              | 257           | 1671                     |
| Toys - 30                | 30              | 235           | 1528                     |
| Toys - 50                | 50              | 291           | 1892                     |
| Toys - 300               | 300             | 199           | 1294                     |
| Toys - 500               | 500             | 264           | 1716                     |

---

**Summary:**

\[
\begin{align*}
\max \quad & \sum_{i \in I} r_i x_i \\
\text{s.t.} \quad & x_i \leq s_i \qquad \forall i \in I \\
                  & x_i \leq d_i \qquad \forall i \in I \\
                  & x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I \\
\end{align*}
\]

Where $I$ is the set of all products as listed above, and all coefficients are as given in the table.