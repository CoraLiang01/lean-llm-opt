##### Sets and Indices

Let $i$ index the products in the order given by ProductName in products.csv:
- Spinach
- Shiitake Mushrooms
- Apples
- Carrots
- Basil
- Potatoes
- Green Beans
- Blueberries
- Oranges
- Watermelons

##### Parameters

Let:
- $v_i$ = Value per unit of product $i$ (from Value column)
- $w_i$ = Weight per unit of product $i$ (from Weight column)
- $C$ = Total stock capacity (from Capacity column in capacity.csv; $C = 875$)

##### Decision Variables

Let $x_i$ = number of units of product $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

##### Mathematical Model

Objective:
\[
\max \quad 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10}
\]

Subject to:
\[
230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
\]

##### Variable/Product Mapping

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

##### Data Used

- Capacity: 875
- Products (in source order):

| ProductName         | Weight | Value |
|---------------------|--------|-------|
| Spinach             | 230    | 64    |
| Shiitake Mushrooms  | 637    | 75    |
| Apples              | 773    | 68    |
| Carrots             | 653    | 11    |
| Basil               | 755    | 91    |
| Potatoes            | 670    | 31    |
| Green Beans         | 505    | 90    |
| Blueberries         | 821    | 56    |
| Oranges             | 83     | 10    |
| Watermelons         | 249    | 24    |