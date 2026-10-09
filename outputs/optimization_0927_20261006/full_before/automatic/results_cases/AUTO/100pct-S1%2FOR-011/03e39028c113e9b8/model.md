Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products as listed below.

Objective:
\[
\max \; 64\,x_{\text{Spinach}} + 75\,x_{\text{Shiitake Mushrooms}} + 68\,x_{\text{Apples}} + 11\,x_{\text{Carrots}} + 91\,x_{\text{Basil}} + 31\,x_{\text{Potatoes}} + 90\,x_{\text{Green Beans}} + 56\,x_{\text{Blueberries}} + 10\,x_{\text{Oranges}} + 24\,x_{\text{Watermelons}}
\]

Subject to:

Stock capacity constraint:
\[
230\,x_{\text{Spinach}} + 637\,x_{\text{Shiitake Mushrooms}} + 773\,x_{\text{Apples}} + 653\,x_{\text{Carrots}} + 755\,x_{\text{Basil}} + 670\,x_{\text{Potatoes}} + 505\,x_{\text{Green Beans}} + 821\,x_{\text{Blueberries}} + 83\,x_{\text{Oranges}} + 249\,x_{\text{Watermelons}} \leq 875
\]

Non-negativity and integrality:
\[
x_{\text{Spinach}},\;
x_{\text{Shiitake Mushrooms}},\;
x_{\text{Apples}},\;
x_{\text{Carrots}},\;
x_{\text{Basil}},\;
x_{\text{Potatoes}},\;
x_{\text{Green Beans}},\;
x_{\text{Blueberries}},\;
x_{\text{Oranges}},\;
x_{\text{Watermelons}}
\in \mathbb{Z}_{\geq 0}
\]

Where:
- Product coefficients in the objective are from the "Value" column.
- Product coefficients in the constraint are from the "Weight" column.
- The right-hand side of the constraint is the "Capacity" from capacity.csv.