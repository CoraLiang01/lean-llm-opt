Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products as listed below.

##### Objective Function

\[
\max \; 64x_{\text{Spinach}} + 75x_{\text{Shiitake Mushrooms}} + 68x_{\text{Apples}} + 11x_{\text{Carrots}} + 91x_{\text{Basil}} + 31x_{\text{Potatoes}} + 90x_{\text{Green Beans}} + 56x_{\text{Blueberries}} + 10x_{\text{Oranges}} + 24x_{\text{Watermelons}}
\]

##### Constraint

\[
230x_{\text{Spinach}}
+ 637x_{\text{Shiitake Mushrooms}}
+ 773x_{\text{Apples}}
+ 653x_{\text{Carrots}}
+ 755x_{\text{Basil}}
+ 670x_{\text{Potatoes}}
+ 505x_{\text{Green Beans}}
+ 821x_{\text{Blueberries}}
+ 83x_{\text{Oranges}}
+ 249x_{\text{Watermelons}}
\leq 875
\]

##### Variable Domains

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for all products } i
\]

##### Product List (in source order):

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

Where:
- The coefficient of $x_i$ in the objective is the "Value" for product $i$.
- The coefficient of $x_i$ in the constraint is the "Weight" for product $i$.
- The right-hand side of the constraint is the "Capacity" from capacity.csv: $875$.

All variables are nonnegative integers.