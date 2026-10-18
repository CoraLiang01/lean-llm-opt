##### Sets and Indices

Let $I = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of shelves (indexed by $i$).

Let $J = \{1,2,\ldots,20\}$ be the set of products (indexed by $j$).

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
j & \text{Value}_j & \text{Weight}_j \\
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

##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on shelf $i$.

##### Objective

\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} \text{Value}_j \cdot x_{ij}
\]

##### Constraints

For each shelf $i \in I$:
\[
\sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]

For all $i \in I$, $j \in J$:
\[
x_{ij} \geq 0,\quad x_{ij} \in \mathbb{Z}
\]

##### Retrieved Information

- Shelf capacities:
  - Shelf 1: 750
  - Shelf 2: 820
  - Shelf 3: 570
  - Shelf 4: 800
  - Shelf 5: 550
  - Shelf 6: 900
  - Shelf 7: 650
  - Shelf 8: 800
  - Shelf 9: 850
  - Shelf 10: 900

- Product values and weights:
  - Product 1: Value 55, Weight 10
  - Product 2: Value 75, Weight 20
  - Product 3: Value 65, Weight 5
  - Product 4: Value 60, Weight 15
  - Product 5: Value 80, Weight 25
  - Product 6: Value 90, Weight 35
  - Product 7: Value 40, Weight 45
  - Product 8: Value 100, Weight 55
  - Product 9: Value 55, Weight 65
  - Product 10: Value 75, Weight 20
  - Product 11: Value 110, Weight 18
  - Product 12: Value 50, Weight 28
  - Product 13: Value 60, Weight 8
  - Product 14: Value 120, Weight 28
  - Product 15: Value 70, Weight 25
  - Product 16: Value 110, Weight 40
  - Product 17: Value 50, Weight 55
  - Product 18: Value 60, Weight 70
  - Product 19: Value 120, Weight 85
  - Product 20: Value 100, Weight 100