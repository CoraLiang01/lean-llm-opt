Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products in the order given below.

#### Objective Function

\[
\max \; 64\,x_{\text{Spinach}} + 75\,x_{\text{Shiitake Mushrooms}} + 68\,x_{\text{Apples}} + 11\,x_{\text{Carrots}} + 91\,x_{\text{Basil}} + 31\,x_{\text{Potatoes}} + 90\,x_{\text{Green Beans}} + 56\,x_{\text{Blueberries}} + 10\,x_{\text{Oranges}} + 24\,x_{\text{Watermelons}}
\]

#### Constraint

\[
230\,x_{\text{Spinach}} + 637\,x_{\text{Shiitake Mushrooms}} + 773\,x_{\text{Apples}} + 653\,x_{\text{Carrots}} + 755\,x_{\text{Basil}} + 670\,x_{\text{Potatoes}} + 505\,x_{\text{Green Beans}} + 821\,x_{\text{Blueberries}} + 83\,x_{\text{Oranges}} + 249\,x_{\text{Watermelons}} \leq 875
\]

#### Variable Domains

\[
x_{\text{Spinach}},\; x_{\text{Shiitake Mushrooms}},\; x_{\text{Apples}},\; x_{\text{Carrots}},\; x_{\text{Basil}},\; x_{\text{Potatoes}},\; x_{\text{Green Beans}},\; x_{\text{Blueberries}},\; x_{\text{Oranges}},\; x_{\text{Watermelons}} \in \mathbb{Z}_{\geq 0}
\]

#### Data Used (in source order)

- Capacity: 875
- Products:

    1. Spinach: Value = 64, Weight = 230
    2. Shiitake Mushrooms: Value = 75, Weight = 637
    3. Apples: Value = 68, Weight = 773
    4. Carrots: Value = 11, Weight = 653
    5. Basil: Value = 91, Weight = 755
    6. Potatoes: Value = 31, Weight = 670
    7. Green Beans: Value = 90, Weight = 505
    8. Blueberries: Value = 56, Weight = 821
    9. Oranges: Value = 10, Weight = 83
    10. Watermelons: Value = 24, Weight = 249