##### Sets and Indices

Let $i$ index the products in the order given in products.csv:
- 1: Spinach
- 2: Shiitake Mushrooms
- 3: Apples
- 4: Carrots
- 5: Basil
- 6: Potatoes
- 7: Green Beans
- 8: Blueberries
- 9: Oranges
- 10: Watermelons

##### Parameters

Let:
- $v_i$ = Value of product $i$ (from the "Value" column)
- $w_i$ = Weight of product $i$ (from the "Weight" column)
- $C$ = Capacity (from capacity.csv, "Capacity" column; value: 875)

##### Decision Variables

- $x_i$ = number of units of product $i$ to order each day, $x_i \in \mathbb{Z}_{\geq 0}$

##### Mathematical Model

Objective:
\[
\max \sum_{i=1}^{10} v_i x_i
\]
where
\[
(v_1, v_2, \ldots, v_{10}) = (64, 75, 68, 11, 91, 31, 90, 56, 10, 24)
\]

Subject to:

Capacity constraint:
\[
\sum_{i=1}^{10} w_i x_i \leq 875
\]
where
\[
(w_1, w_2, \ldots, w_{10}) = (230, 637, 773, 653, 755, 670, 505, 821, 83, 249)
\]

Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 10
\]

##### Complete Model (with data):

\[
\begin{align*}
\max\quad & 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10} \\
\text{s.t.}\quad & 230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875 \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad i=1,\ldots,10
\end{align*}
\]

Where:
- $x_1$: Spinach
- $x_2$: Shiitake Mushrooms
- $x_3$: Apples
- $x_4$: Carrots
- $x_5$: Basil
- $x_6$: Potatoes
- $x_7$: Green Beans
- $x_8$: Blueberries
- $x_9$: Oranges
- $x_{10}$: Watermelons