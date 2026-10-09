Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products in the order they appear in products.csv.

##### Objective Function

\[
\max \; 64x_{\text{Spinach}} + 75x_{\text{Shiitake Mushrooms}} + 68x_{\text{Apples}} + 11x_{\text{Carrots}} + 91x_{\text{Basil}} + 31x_{\text{Potatoes}} + 90x_{\text{Green Beans}} + 56x_{\text{Blueberries}} + 10x_{\text{Oranges}} + 24x_{\text{Watermelons}}
\]

##### Subject to

\[
230x_{\text{Spinach}} + 637x_{\text{Shiitake Mushrooms}} + 773x_{\text{Apples}} + 653x_{\text{Carrots}} + 755x_{\text{Basil}} + 670x_{\text{Potatoes}} + 505x_{\text{Green Beans}} + 821x_{\text{Blueberries}} + 83x_{\text{Oranges}} + 249x_{\text{Watermelons}} \leq 875
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{Spinach}, \text{Shiitake Mushrooms}, \text{Apples}, \text{Carrots}, \text{Basil}, \text{Potatoes}, \text{Green Beans}, \text{Blueberries}, \text{Oranges}, \text{Watermelons}\}
\]

##### Where

- The coefficients in the objective are the "Value" for each product from products.csv.
- The coefficients in the constraint are the "Weight" for each product from products.csv.
- The right-hand side of the constraint is the "Capacity" from capacity.csv.
- All variables are nonnegative integers.