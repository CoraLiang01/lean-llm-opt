Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products as listed in products.csv.

Objective:
$$
\max \; 64x_{\text{Spinach}} + 75x_{\text{Shiitake Mushrooms}} + 68x_{\text{Apples}} + 11x_{\text{Carrots}} + 91x_{\text{Basil}} + 31x_{\text{Potatoes}} + 90x_{\text{Green Beans}} + 56x_{\text{Blueberries}} + 10x_{\text{Oranges}} + 24x_{\text{Watermelons}}
$$

Subject to:

Capacity constraint:
$$
230x_{\text{Spinach}} + 637x_{\text{Shiitake Mushrooms}} + 773x_{\text{Apples}} + 653x_{\text{Carrots}} + 755x_{\text{Basil}} + 670x_{\text{Potatoes}} + 505x_{\text{Green Beans}} + 821x_{\text{Blueberries}} + 83x_{\text{Oranges}} + 249x_{\text{Watermelons}} \leq 875
$$

Nonnegativity and integrality:
$$
x_{\text{Spinach}},\; x_{\text{Shiitake Mushrooms}},\; x_{\text{Apples}},\; x_{\text{Carrots}},\; x_{\text{Basil}},\; x_{\text{Potatoes}},\; x_{\text{Green Beans}},\; x_{\text{Blueberries}},\; x_{\text{Oranges}},\; x_{\text{Watermelons}} \in \mathbb{Z}_{\geq 0}
$$

Where:
- The coefficients in the objective are from the "Value" column in products.csv.
- The coefficients in the constraint are from the "Weight" column in products.csv.
- The right-hand side of the constraint is the "Capacity" from capacity.csv.