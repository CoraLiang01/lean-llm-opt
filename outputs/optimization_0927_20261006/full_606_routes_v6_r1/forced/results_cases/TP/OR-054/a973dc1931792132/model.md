##### Decision Variables

Let $x_{ij} \geq 0$ and integer: number of units of product $j$ placed on shelf $i$.

Where:
- $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (ShelfID)
- $j \in \{1,2,\ldots,20\}$ (ProductName)

##### Parameters

- Shelf capacities (from capacity.csv):

\[
\begin{align*}
\text{Capacity}_1 &= 750 \\
\text{Capacity}_2 &= 820 \\
\text{Capacity}_3 &= 570 \\
\text{Capacity}_4 &= 800 \\
\text{Capacity}_5 &= 550 \\
\text{Capacity}_6 &= 900 \\
\text{Capacity}_7 &= 650 \\
\text{Capacity}_8 &= 800 \\
\text{Capacity}_9 &= 850 \\
\text{Capacity}_{10} &= 900 \\
\end{align*}
\]

- Product values and weights (from products.csv):

\[
\begin{array}{cccc}
\text{ProductName} & \text{Value}_j & \text{Weight}_j \\
1 & 55 & 10 \\
2 & 75 & 20 \\
3 & 65 & 5 \\
4 & 60 & 15 \\
5 & 80 & 25 \\
6 & 90 & 35 \\
7 & 40 & 45 \\
8 & 100 & 55 \\
9 & 55 & 65 \\
10 & 75 & 20 \\
11 & 110 & 18 \\
12 & 50 & 28 \\
13 & 60 & 8 \\
14 & 120 & 28 \\
15 & 70 & 25 \\
16 & 110 & 40 \\
17 & 50 & 55 \\
18 & 60 & 70 \\
19 & 120 & 85 \\
20 & 100 & 100 \\
\end{array}
\]

##### Objective Function

\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} \text{Value}_j \cdot x_{ij}
\]

##### Constraints

For each shelf $i$:
\[
\sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i \qquad \forall i \in \{1,\ldots,10\}
\]

Non-negativity and integrality:
\[
x_{ij} \geq 0,\quad x_{ij} \in \mathbb{Z} \qquad \forall i \in \{1,\ldots,10\},\; j \in \{1,\ldots,20\}
\]

##### Retrieved Information

- Shelf capacities:
  - 1: 750
  - 2: 820
  - 3: 570
  - 4: 800
  - 5: 550
  - 6: 900
  - 7: 650
  - 8: 800
  - 9: 850
  - 10: 900

- Product values and weights:
  - 1: value 55, weight 10
  - 2: value 75, weight 20
  - 3: value 65, weight 5
  - 4: value 60, weight 15
  - 5: value 80, weight 25
  - 6: value 90, weight 35
  - 7: value 40, weight 45
  - 8: value 100, weight 55
  - 9: value 55, weight 65
  - 10: value 75, weight 20
  - 11: value 110, weight 18
  - 12: value 50, weight 28
  - 13: value 60, weight 8
  - 14: value 120, weight 28
  - 15: value 70, weight 25
  - 16: value 110, weight 40
  - 17: value 50, weight 55
  - 18: value 60, weight 70
  - 19: value 120, weight 85
  - 20: value 100, weight 100