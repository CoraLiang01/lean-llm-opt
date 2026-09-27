Let $x_{ij}$ be the number of units of product $j$ placed on shelf $i$, where $i \in \{1,2,\ldots,10\}$ (ShelfID from capacity.csv) and $j \in \{1,2,\ldots,20\}$ (ProductName from products.csv).

Parameters:

- Shelf capacities (from capacity.csv):

\[
\begin{align*}
&\text{Shelf 1: } C_1 = 750 \\
&\text{Shelf 2: } C_2 = 820 \\
&\text{Shelf 3: } C_3 = 570 \\
&\text{Shelf 4: } C_4 = 800 \\
&\text{Shelf 5: } C_5 = 550 \\
&\text{Shelf 6: } C_6 = 900 \\
&\text{Shelf 7: } C_7 = 650 \\
&\text{Shelf 8: } C_8 = 800 \\
&\text{Shelf 9: } C_9 = 850 \\
&\text{Shelf 10: } C_{10} = 900 \\
\end{align*}
\]

- Product values and weights (from products.csv):

\[
\begin{align*}
&\text{Product 1: } v_1 = 55, \quad w_1 = 10 \\
&\text{Product 2: } v_2 = 75, \quad w_2 = 20 \\
&\text{Product 3: } v_3 = 65, \quad w_3 = 5 \\
&\text{Product 4: } v_4 = 60, \quad w_4 = 15 \\
&\text{Product 5: } v_5 = 80, \quad w_5 = 25 \\
&\text{Product 6: } v_6 = 90, \quad w_6 = 35 \\
&\text{Product 7: } v_7 = 40, \quad w_7 = 45 \\
&\text{Product 8: } v_8 = 100, \quad w_8 = 55 \\
&\text{Product 9: } v_9 = 55, \quad w_9 = 65 \\
&\text{Product 10: } v_{10} = 75, \quad w_{10} = 20 \\
&\text{Product 11: } v_{11} = 110, \quad w_{11} = 18 \\
&\text{Product 12: } v_{12} = 50, \quad w_{12} = 28 \\
&\text{Product 13: } v_{13} = 60, \quad w_{13} = 8 \\
&\text{Product 14: } v_{14} = 120, \quad w_{14} = 28 \\
&\text{Product 15: } v_{15} = 70, \quad w_{15} = 25 \\
&\text{Product 16: } v_{16} = 110, \quad w_{16} = 40 \\
&\text{Product 17: } v_{17} = 50, \quad w_{17} = 55 \\
&\text{Product 18: } v_{18} = 60, \quad w_{18} = 70 \\
&\text{Product 19: } v_{19} = 120, \quad w_{19} = 85 \\
&\text{Product 20: } v_{20} = 100, \quad w_{20} = 100 \\
\end{align*}
\]

Model:

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \, x_{ij}
\]

Subject to (for each shelf $i$):

\[
\sum_{j=1}^{20} w_j \, x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\; j = 1,\ldots,20
\]

Where:

- $x_{ij}$: Number of units of product $j$ placed on shelf $i$
- $v_j$: Value of product $j$ (see above)
- $w_j$: Weight of product $j$ (see above)
- $C_i$: Capacity of shelf $i$ (see above)