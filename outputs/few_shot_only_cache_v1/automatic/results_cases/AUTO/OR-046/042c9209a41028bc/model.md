Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products in the order given in products.csv.

Parameters (from products.csv, in source order):

\[
\begin{array}{lll}
\text{ProductName} & \text{Weight} & \text{Value} \\
\hline
\text{Spinach} & 230 & 64 \\
\text{Shiitake Mushrooms} & 637 & 75 \\
\text{Apples} & 773 & 68 \\
\text{Carrots} & 653 & 11 \\
\text{Basil} & 755 & 91 \\
\text{Potatoes} & 670 & 31 \\
\text{Green Beans} & 505 & 90 \\
\text{Blueberries} & 821 & 56 \\
\text{Oranges} & 83 & 10 \\
\text{Watermelons} & 249 & 24 \\
\end{array}
\]

Total stock capacity (from capacity.csv):

\[
\text{Capacity} = 875
\]

Model:

\[
\textbf{Objective:} \quad \max \left( 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10} \right)
\]

\[
\textbf{Subject to:}
\]

\[
230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
\]

Where the mapping of $i$ to products is as follows (in source order):

\[
\begin{align*}
x_1 &: \text{Spinach} \\
x_2 &: \text{Shiitake Mushrooms} \\
x_3 &: \text{Apples} \\
x_4 &: \text{Carrots} \\
x_5 &: \text{Basil} \\
x_6 &: \text{Potatoes} \\
x_7 &: \text{Green Beans} \\
x_8 &: \text{Blueberries} \\
x_9 &: \text{Oranges} \\
x_{10} &: \text{Watermelons} \\
\end{align*}
\]