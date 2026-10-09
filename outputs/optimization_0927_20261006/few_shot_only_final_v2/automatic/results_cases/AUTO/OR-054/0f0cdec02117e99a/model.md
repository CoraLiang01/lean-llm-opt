Let  
- $i$ index shelves, with $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (from "capacity.csv", column ShelfID)  
- $j$ index products, with $j \in \{1,2,\ldots,20\}$ (from "products.csv", column ProductName)  
- $x_{ij}$ = number of units of product $j$ placed on shelf $i$ ($x_{ij} \in \mathbb{Z}_{\geq 0}$)  
- $c_i$ = capacity of shelf $i$ (from "capacity.csv", column Capacity)  
- $v_j$ = value of product $j$ (from "products.csv", column Value)  
- $w_j$ = weight of product $j$ (from "products.csv", column Weight)  

**Parameters (from data):**

Shelf capacities ($c_i$):  
\[
\begin{align*}
c_1 &= 750 \\
c_2 &= 820 \\
c_3 &= 570 \\
c_4 &= 800 \\
c_5 &= 550 \\
c_6 &= 900 \\
c_7 &= 650 \\
c_8 &= 800 \\
c_9 &= 850 \\
c_{10} &= 900 \\
\end{align*}
\]

Product values ($v_j$) and weights ($w_j$):  
\[
\begin{array}{c|c|c}
\text{ProductName } (j) & v_j & w_j \\
\hline
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

**Mathematical Model:**

**Decision Variables:**  
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
\]

**Objective Function:**  
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j\, x_{ij}
\]

**Constraints:**  
For each shelf $i$:
\[
\sum_{j=1}^{20} w_j\, x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
\]

**Variable Domains:**  
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
\]

**Explicitly, the model is:**

\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j\, x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{20} w_j\, x_{i j} \leq c_i \qquad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\ j = 1,\ldots,20
\end{align*}
\]

Where the parameters $v_j$, $w_j$, and $c_i$ are as listed above, using the original identifiers and coefficients from the data.