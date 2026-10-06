##### Sets and Indices

Let $i$ index the products as listed in the original order of products.csv:
1. Spinach
2. Shiitake Mushrooms
3. Apples
4. Carrots
5. Basil
6. Potatoes
7. Green Beans
8. Blueberries
9. Oranges
10. Watermelons

##### Parameters

- $v_i$: Value per unit of product $i$ (from Value column)
- $w_i$: Weight per unit of product $i$ (from Weight column)
- $C$: Total stock capacity (from Capacity column in capacity.csv)

##### Decision Variables

- $x_i$: Number of units of product $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

##### Objective Function

Maximize total benefit:
$$
\max \sum_{i=1}^{10} v_i x_i
$$

That is,
$$
\max \big(64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10}\big)
$$

##### Constraints

Total weight of ordered products cannot exceed capacity:
$$
230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875
$$

Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,10
$$

##### Complete Model

\[
\begin{align*}
\max\ & 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10} \\
\text{s.t.}\quad & 230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875 \\
& x_i \in \mathbb{Z}_{\geq 0},\quad \forall i = 1,\ldots,10
\end{align*}
\]

Where the products correspond to the following order:
1. Spinach
2. Shiitake Mushrooms
3. Apples
4. Carrots
5. Basil
6. Potatoes
7. Green Beans
8. Blueberries
9. Oranges
10. Watermelons