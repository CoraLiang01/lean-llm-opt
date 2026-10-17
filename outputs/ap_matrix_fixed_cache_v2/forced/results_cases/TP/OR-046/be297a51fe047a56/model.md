Let $x_i \geq 0$ be the number of units of product $i$ to order each day, for each product $i$ in the set:
\[
\text{Products} = \{\text{Spinach},\ \text{Shiitake Mushrooms},\ \text{Apples},\ \text{Carrots},\ \text{Basil},\ \text{Potatoes},\ \text{Green Beans},\ \text{Blueberries},\ \text{Oranges},\ \text{Watermelons}\}
\]

Parameters (from products.csv, in source order):

\[
\begin{array}{lll}
\text{Product} & \text{Weight}_i & \text{Value}_i \\
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

Mathematical Model:

Maximize total benefit:
\[
\max\ 64x_{\text{Spinach}} + 75x_{\text{Shiitake Mushrooms}} + 68x_{\text{Apples}} + 11x_{\text{Carrots}} + 91x_{\text{Basil}} + 31x_{\text{Potatoes}} + 90x_{\text{Green Beans}} + 56x_{\text{Blueberries}} + 10x_{\text{Oranges}} + 24x_{\text{Watermelons}}
\]

Subject to the stock capacity constraint:
\[
230x_{\text{Spinach}} + 637x_{\text{Shiitake Mushrooms}} + 773x_{\text{Apples}} + 653x_{\text{Carrots}} + 755x_{\text{Basil}} + 670x_{\text{Potatoes}} + 505x_{\text{Green Beans}} + 821x_{\text{Blueberries}} + 83x_{\text{Oranges}} + 249x_{\text{Watermelons}} \leq 875
\]

Non-negativity:
\[
x_i \geq 0 \quad \text{for all products } i
\]