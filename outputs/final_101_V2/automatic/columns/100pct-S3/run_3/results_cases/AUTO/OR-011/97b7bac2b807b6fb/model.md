Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products as listed in products.csv.

Objective:
\[
\max \; 64\,x_{\text{Spinach}} + 75\,x_{\text{Shiitake Mushrooms}} + 68\,x_{\text{Apples}} + 11\,x_{\text{Carrots}} + 91\,x_{\text{Basil}} + 31\,x_{\text{Potatoes}} + 90\,x_{\text{Green Beans}} + 56\,x_{\text{Blueberries}} + 10\,x_{\text{Oranges}} + 24\,x_{\text{Watermelons}}
\]

Subject to:

\[
230\,x_{\text{Spinach}} + 637\,x_{\text{Shiitake Mushrooms}} + 773\,x_{\text{Apples}} + 653\,x_{\text{Carrots}} + 755\,x_{\text{Basil}} + 670\,x_{\text{Potatoes}} + 505\,x_{\text{Green Beans}} + 821\,x_{\text{Blueberries}} + 83\,x_{\text{Oranges}} + 249\,x_{\text{Watermelons}} \leq 875
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{Spinach}, \text{Shiitake Mushrooms}, \text{Apples}, \text{Carrots}, \text{Basil}, \text{Potatoes}, \text{Green Beans}, \text{Blueberries}, \text{Oranges}, \text{Watermelons}\}
\]

Where:
- $x_i$ = number of units of product $i$ to order each day (nonnegative integer)
- The coefficient for each $x_i$ in the objective is the Value from products.csv
- The coefficient for each $x_i$ in the constraint is the Weight from products.csv
- The right-hand side of the constraint is the Capacity from capacity.csv (875)