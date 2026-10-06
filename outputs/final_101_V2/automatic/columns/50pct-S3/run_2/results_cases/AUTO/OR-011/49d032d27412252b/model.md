##### Sets and Indices

Let $i$ index the products as listed in products.csv:
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

For each product $i$:
- $v_i$ = Value of product $i$ (from products.csv)
- $w_i$ = Weight of product $i$ (from products.csv)

Let $C$ = Capacity (from capacity.csv), i.e., $C = 875$

##### Decision Variables

For each product $i$:
- $x_i$ = number of units of product $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

##### Mathematical Model

Objective:
\[
\max \sum_{i} v_i x_i
\]
where $v_i$ and $x_i$ are as defined above.

Subject to:

\[
\sum_{i} w_i x_i \leq C
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

##### Data

| ProductName           | $w_i$ (Weight) | $v_i$ (Value) |
|-----------------------|----------------|---------------|
| Spinach               | 230            | 64            |
| Shiitake Mushrooms    | 637            | 75            |
| Apples                | 773            | 68            |
| Carrots               | 653            | 11            |
| Basil                 | 755            | 91            |
| Potatoes              | 670            | 31            |
| Green Beans           | 505            | 90            |
| Blueberries           | 821            | 56            |
| Oranges               | 83             | 10            |
| Watermelons           | 249            | 24            |

Capacity: $C = 875$

##### Complete Model (with data):

\[
\max \left(
64\,x_{\text{Spinach}} +
75\,x_{\text{Shiitake Mushrooms}} +
68\,x_{\text{Apples}} +
11\,x_{\text{Carrots}} +
91\,x_{\text{Basil}} +
31\,x_{\text{Potatoes}} +
90\,x_{\text{Green Beans}} +
56\,x_{\text{Blueberries}} +
10\,x_{\text{Oranges}} +
24\,x_{\text{Watermelons}}
\right)
\]

subject to

\[
230\,x_{\text{Spinach}} +
637\,x_{\text{Shiitake Mushrooms}} +
773\,x_{\text{Apples}} +
653\,x_{\text{Carrots}} +
755\,x_{\text{Basil}} +
670\,x_{\text{Potatoes}} +
505\,x_{\text{Green Beans}} +
821\,x_{\text{Blueberries}} +
83\,x_{\text{Oranges}} +
249\,x_{\text{Watermelons}}
\leq 875
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]