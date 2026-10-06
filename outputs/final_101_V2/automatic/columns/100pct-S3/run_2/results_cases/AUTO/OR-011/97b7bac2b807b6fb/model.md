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
- $C$ = Capacity (from the "Capacity" column in capacity.csv)

From the data:
- $v = [64, 75, 68, 11, 91, 31, 90, 56, 10, 24]$
- $w = [230, 637, 773, 653, 755, 670, 505, 821, 83, 249]$
- $C = 875$

##### Decision Variables

Let $x_i$ = number of units of product $i$ to order each day, $x_i \in \mathbb{Z}_{\geq 0}$

##### Mathematical Model

Objective:
\[
\max \sum_{i=1}^{10} v_i x_i = 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10}
\]

Subject to:
\[
\sum_{i=1}^{10} w_i x_i = 230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
\]

##### Data Used

capacity.csv:
- previous_period_capacity: 991
- Capacity: 875

products.csv (in order):
- Spinach: Weight 230, Value 64
- Shiitake Mushrooms: Weight 637, Value 75
- Apples: Weight 773, Value 68
- Carrots: Weight 653, Value 11
- Basil: Weight 755, Value 91
- Potatoes: Weight 670, Value 31
- Green Beans: Weight 505, Value 90
- Blueberries: Weight 821, Value 56
- Oranges: Weight 83, Value 10
- Watermelons: Weight 249, Value 24