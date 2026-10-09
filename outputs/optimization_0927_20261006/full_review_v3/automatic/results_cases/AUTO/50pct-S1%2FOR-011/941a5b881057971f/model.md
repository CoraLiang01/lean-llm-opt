##### Decision Variables

Let $x_i$ be the number of units of product $i$ to order each day, for each product $i$ listed below. All $x_i$ are nonnegative integers.

##### Parameters

- For each product $i$:
    - $v_i$ = Value (from "Value" column)
    - $w_i$ = Weight (from "Weight" column)
- $C$ = 875 (from "Capacity" in capacity.csv)

Products and their parameters (in source order):

| ProductName           | $v_i$ | $w_i$ |
|-----------------------|-------|-------|
| Spinach               | 64    | 230   |
| Shiitake Mushrooms    | 75    | 637   |
| Apples                | 68    | 773   |
| Carrots               | 11    | 653   |
| Basil                 | 91    | 755   |
| Potatoes              | 31    | 670   |
| Green Beans           | 90    | 505   |
| Blueberries           | 56    | 821   |
| Oranges               | 10    | 83    |
| Watermelons           | 24    | 249   |

##### Mathematical Model

Objective:
$$
\max \; 64x_{\text{Spinach}} + 75x_{\text{Shiitake Mushrooms}} + 68x_{\text{Apples}} + 11x_{\text{Carrots}} + 91x_{\text{Basil}} + 31x_{\text{Potatoes}} + 90x_{\text{Green Beans}} + 56x_{\text{Blueberries}} + 10x_{\text{Oranges}} + 24x_{\text{Watermelons}}
$$

Subject to:
$$
230x_{\text{Spinach}} + 637x_{\text{Shiitake Mushrooms}} + 773x_{\text{Apples}} + 653x_{\text{Carrots}} + 755x_{\text{Basil}} + 670x_{\text{Potatoes}} + 505x_{\text{Green Beans}} + 821x_{\text{Blueberries}} + 83x_{\text{Oranges}} + 249x_{\text{Watermelons}} \leq 875
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

Where $i$ ranges over the following products (in source order):

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

##### Retrieved Information

- Capacity: 875
- Products (ProductName, Weight, Value):

    - Spinach, 230, 64
    - Shiitake Mushrooms, 637, 75
    - Apples, 773, 68
    - Carrots, 653, 11
    - Basil, 755, 91
    - Potatoes, 670, 31
    - Green Beans, 505, 90
    - Blueberries, 821, 56
    - Oranges, 83, 10
    - Watermelons, 249, 24

All variables $x_i$ are nonnegative integers.