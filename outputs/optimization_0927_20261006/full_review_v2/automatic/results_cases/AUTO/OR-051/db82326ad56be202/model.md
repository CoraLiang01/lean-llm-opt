Let $x_{ij}$ be the number of units of coffee product $j$ to be placed in cabinet $i$. All $x_{ij}$ are nonnegative integers.

**Indices:**
- $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (CabinetID)
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
$\}$ (ProductName)

**Parameters:**

Cabinet capacities:
\[
\begin{align*}
\text{Capacity}_1 &= 400 \\
\text{Capacity}_2 &= 600 \\
\text{Capacity}_3 &= 500 \\
\text{Capacity}_4 &= 700 \\
\text{Capacity}_5 &= 450 \\
\text{Capacity}_6 &= 650 \\
\text{Capacity}_7 &= 550 \\
\text{Capacity}_8 &= 750 \\
\text{Capacity}_9 &= 480 \\
\text{Capacity}_{10} &= 520 \\
\end{align*}
\]

Product values and weights:
\[
\begin{array}{lll}
\text{ProductName} & \text{Value}_j & \text{Weight}_j \\
\hline
\text{Espresso Beans} & 100 & 1.0 \\
\text{Colombian Roast} & 150 & 1.5 \\
\text{Arabica Blend} & 80 & 1.2 \\
\text{French Roast} & 120 & 1.3 \\
\text{Italian Roast} & 130 & 1.4 \\
\text{House Blend} & 110 & 1.1 \\
\text{Sumatra Coffee} & 160 & 1.8 \\
\text{Mocha Java} & 90 & 1.2 \\
\text{Hazelnut Flavor} & 95 & 1.0 \\
\text{Caramel Blend} & 105 & 1.3 \\
\text{Vanilla Flavor} & 85 & 1.2 \\
\text{Cappuccino Mix} & 140 & 1.5 \\
\text{Pumpkin Spice} & 75 & 1.1 \\
\text{Decaf Roast} & 60 & 1.0 \\
\text{Organic Roast} & 170 & 1.6 \\
\text{Cold Brew} & 115 & 1.4 \\
\text{Peruvian Blend} & 155 & 1.7 \\
\text{Kenyan AA} & 125 & 1.3 \\
\end{array}
\]

---

### Mathematical Model

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j} \text{Value}_j \cdot x_{ij}
\]

**Subject to:**

For each cabinet $i$:
\[
\sum_{j} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
\]

**Variable domains:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

---

**Data used:**

Cabinet capacities (in order):

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

Product values and weights (in order):

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