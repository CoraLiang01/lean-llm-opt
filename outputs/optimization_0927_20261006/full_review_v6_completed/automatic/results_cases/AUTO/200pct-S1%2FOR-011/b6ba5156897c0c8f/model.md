Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products in the order they appear in products.csv.

Parameters (from products.csv, in source order):

\[
\begin{array}{lll}
\text{ProductName} & \text{Value} & \text{Weight} \\
\hline
\text{Spinach} & 64 & 230 \\
\text{Shiitake Mushrooms} & 75 & 637 \\
\text{Apples} & 68 & 773 \\
\text{Carrots} & 11 & 653 \\
\text{Basil} & 91 & 755 \\
\text{Potatoes} & 31 & 670 \\
\text{Green Beans} & 90 & 505 \\
\text{Blueberries} & 56 & 821 \\
\text{Oranges} & 10 & 83 \\
\text{Watermelons} & 24 & 249 \\
\end{array}
\]

Total stock capacity (from capacity.csv):

\[
\text{Capacity} = 875
\]

Model:

Objective:
\[
\max \; 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10}
\]

Subject to:
\[
230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,2,\ldots,10
\]

Where:
- $x_1$ = units of Spinach to order
- $x_2$ = units of Shiitake Mushrooms to order
- $x_3$ = units of Apples to order
- $x_4$ = units of Carrots to order
- $x_5$ = units of Basil to order
- $x_6$ = units of Potatoes to order
- $x_7$ = units of Green Beans to order
- $x_8$ = units of Blueberries to order
- $x_9$ = units of Oranges to order
- $x_{10}$ = units of Watermelons to order