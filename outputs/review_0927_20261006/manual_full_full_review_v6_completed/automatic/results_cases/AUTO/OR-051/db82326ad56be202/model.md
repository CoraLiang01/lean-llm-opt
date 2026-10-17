Let $x_{ij}$ be the number of units of coffee product $j$ to be placed in cabinet $i$. All $x_{ij}$ are integer and $\geq 0$.

Let:
- $i$ index cabinets, with CabinetID from the data.
- $j$ index products, with ProductName from the data.
- $c_i$ = Capacity of cabinet $i$.
- $v_j$ = Value of product $j$.
- $w_j$ = Weight of product $j$.

**Objective:**
\[
\max \sum_{i \in \{\text{1,2,3,4,5,6,7,8,9,10}\}} \sum_{j \in \mathcal{P}} v_j \cdot x_{ij}
\]
where $\mathcal{P}$ is the set of all ProductNames below.

**Subject to:**

For each cabinet $i$:
\[
\sum_{j \in \mathcal{P}} w_j \cdot x_{ij} \leq c_i
\]

For all $i, j$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

**Data:**

Cabinets and Capacities (from capacity.csv, in source order):

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

Products (from products.csv, in source order):

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

**Decision variables:**
\[
x_{ij} = \text{number of units of product } j \text{ in cabinet } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
\]

**Complete Model:**

\[
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j=1}^{18} v_j x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{18} w_j x_{ij} \leq c_i, \quad \forall i \in \{1,2,\ldots,10\} \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i, j
\end{align*}
\]

Where the mapping of $i$ and $j$ to CabinetID and ProductName, and the values of $c_i$, $v_j$, $w_j$ are as listed above.