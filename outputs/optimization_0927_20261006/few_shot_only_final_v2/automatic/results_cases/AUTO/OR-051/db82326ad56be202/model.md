**Sets:**
- $i \in \{\text{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}\}$ (CabinetID from "capacity.csv")
- $j \in \{$
    Espresso Beans, 
    Colombian Roast, 
    Arabica Blend, 
    French Roast, 
    Italian Roast, 
    House Blend, 
    Sumatra Coffee, 
    Mocha Java, 
    Hazelnut Flavor, 
    Caramel Blend, 
    Vanilla Flavor, 
    Cappuccino Mix, 
    Pumpkin Spice, 
    Decaf Roast, 
    Organic Roast, 
    Cold Brew, 
    Peruvian Blend, 
    Kenyan AA
$\}$ (ProductName from "products.csv")

**Parameters:**
- $C_i$ = Capacity of cabinet $i$ (from "capacity.csv")
- $v_j$ = Value of product $j$ (from "products.csv")
- $w_j$ = Weight of product $j$ (from "products.csv")

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ placed in cabinet $i$

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{18} v_j \cdot x_{ij}
\]

**Subject to:**

For each cabinet $i$:
\[
\sum_{j=1}^{18} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
\]

For all $i, j$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

**Parameter Values (from source, in order):**

- Cabinet capacities:
    - $C_1 = 400$
    - $C_2 = 600$
    - $C_3 = 500$
    - $C_4 = 700$
    - $C_5 = 450$
    - $C_6 = 650$
    - $C_7 = 550$
    - $C_8 = 750$
    - $C_9 = 480$
    - $C_{10} = 520$

- Product values and weights:

| $j$ | ProductName           | $v_j$ | $w_j$ |
|-----|-----------------------|-------|-------|
| 1   | Espresso Beans        | 100   | 1.0   |
| 2   | Colombian Roast       | 150   | 1.5   |
| 3   | Arabica Blend         | 80    | 1.2   |
| 4   | French Roast          | 120   | 1.3   |
| 5   | Italian Roast         | 130   | 1.4   |
| 6   | House Blend           | 110   | 1.1   |
| 7   | Sumatra Coffee        | 160   | 1.8   |
| 8   | Mocha Java            | 90    | 1.2   |
| 9   | Hazelnut Flavor       | 95    | 1.0   |
| 10  | Caramel Blend         | 105   | 1.3   |
| 11  | Vanilla Flavor        | 85    | 1.2   |
| 12  | Cappuccino Mix        | 140   | 1.5   |
| 13  | Pumpkin Spice         | 75    | 1.1   |
| 14  | Decaf Roast           | 60    | 1.0   |
| 15  | Organic Roast         | 170   | 1.6   |
| 16  | Cold Brew             | 115   | 1.4   |
| 17  | Peruvian Blend        | 155   | 1.7   |
| 18  | Kenyan AA             | 125   | 1.3   |

**Complete Model:**

\[
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j=1}^{18} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{18} w_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\; j = 1,\ldots,18
\end{align*}
\]

Where $v_j$ and $w_j$ are as listed above, and $C_i$ are the cabinet capacities as listed above.