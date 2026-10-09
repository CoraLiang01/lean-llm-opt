**Sets and Indices:**
- Let $i$ index cabinets, with CabinetID from "capacity.csv".
- Let $j$ index products, with ProductName from "products.csv".

**Parameters:**
- $C_i$: Capacity of cabinet $i$ (from "capacity.csv").
- $v_j$: Value of product $j$ (from "products.csv").
- $w_j$: Weight of product $j$ (from "products.csv").

**Decision Variables:**
- $x_{ij}$: Number of units of product $j$ placed in cabinet $i$, $x_{ij} \in \mathbb{Z}_{\geq 0}$.

**Data:**

From "capacity.csv":

| CabinetID | Capacity |
|-----------|----------|
| 1         | 400      |
| 2         | 600      |
| 3         | 500      |
| 4         | 700      |
| 5         | 450      |
| 6         | 650      |
| 7         | 550      |
| 8         | 750      |
| 9         | 480      |
| 10        | 520      |

From "products.csv":

| ProductName         | Value | Weight |
|---------------------|-------|--------|
| Espresso Beans      | 100   | 1.0    |
| Colombian Roast     | 150   | 1.5    |
| Arabica Blend       | 80    | 1.2    |
| French Roast        | 120   | 1.3    |
| Italian Roast       | 130   | 1.4    |
| House Blend         | 110   | 1.1    |
| Sumatra Coffee      | 160   | 1.8    |
| Mocha Java          | 90    | 1.2    |
| Hazelnut Flavor     | 95    | 1.0    |
| Caramel Blend       | 105   | 1.3    |
| Vanilla Flavor      | 85    | 1.2    |
| Cappuccino Mix      | 140   | 1.5    |
| Pumpkin Spice       | 75    | 1.1    |
| Decaf Roast         | 60    | 1.0    |
| Organic Roast       | 170   | 1.6    |
| Cold Brew           | 115   | 1.4    |
| Peruvian Blend      | 155   | 1.7    |
| Kenyan AA           | 125   | 1.3    |

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{all products}\}} v_j \cdot x_{ij}
\]

**Subject to:**

For each cabinet $i$ (CabinetID $= 1,\ldots,10$):
\[
\sum_{j} w_j \cdot x_{ij} \leq C_i
\]

For all $i, j$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

**Explicitly, using the data:**

Let $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (CabinetID), $j$ as below.

- $C_1 = 400$, $C_2 = 600$, $C_3 = 500$, $C_4 = 700$, $C_5 = 450$, $C_6 = 650$, $C_7 = 550$, $C_8 = 750$, $C_9 = 480$, $C_{10} = 520$
- $v_j$, $w_j$ as in the table above for each ProductName.

**Full Model:**

\[
\max \sum_{i=1}^{10} \Bigg(
100\,x_{i,\text{Espresso Beans}} + 150\,x_{i,\text{Colombian Roast}} + 80\,x_{i,\text{Arabica Blend}} + 120\,x_{i,\text{French Roast}} + 130\,x_{i,\text{Italian Roast}} + 110\,x_{i,\text{House Blend}} + 160\,x_{i,\text{Sumatra Coffee}} + 90\,x_{i,\text{Mocha Java}} + 95\,x_{i,\text{Hazelnut Flavor}} + 105\,x_{i,\text{Caramel Blend}} + 85\,x_{i,\text{Vanilla Flavor}} + 140\,x_{i,\text{Cappuccino Mix}} + 75\,x_{i,\text{Pumpkin Spice}} + 60\,x_{i,\text{Decaf Roast}} + 170\,x_{i,\text{Organic Roast}} + 115\,x_{i,\text{Cold Brew}} + 155\,x_{i,\text{Peruvian Blend}} + 125\,x_{i,\text{Kenyan AA}}
\Bigg)
\]

Subject to, for each $i = 1,\ldots,10$:

\[
\begin{align*}
&1.0\,x_{i,\text{Espresso Beans}} + 1.5\,x_{i,\text{Colombian Roast}} + 1.2\,x_{i,\text{Arabica Blend}} + 1.3\,x_{i,\text{French Roast}} + 1.4\,x_{i,\text{Italian Roast}} + 1.1\,x_{i,\text{House Blend}} + 1.8\,x_{i,\text{Sumatra Coffee}} + 1.2\,x_{i,\text{Mocha Java}} \\
&\quad + 1.0\,x_{i,\text{Hazelnut Flavor}} + 1.3\,x_{i,\text{Caramel Blend}} + 1.2\,x_{i,\text{Vanilla Flavor}} + 1.5\,x_{i,\text{Cappuccino Mix}} + 1.1\,x_{i,\text{Pumpkin Spice}} + 1.0\,x_{i,\text{Decaf Roast}} \\
&\quad + 1.6\,x_{i,\text{Organic Roast}} + 1.4\,x_{i,\text{Cold Brew}} + 1.7\,x_{i,\text{Peruvian Blend}} + 1.3\,x_{i,\text{Kenyan AA}} \leq C_i
\end{align*}
\]

For all $i = 1,\ldots,10$, $j$ as above:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]