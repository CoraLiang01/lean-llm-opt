Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products in the order they appear in the data.

**Objective:**
\[
\max \; 64x_{\text{Spinach}} + 75x_{\text{Shiitake Mushrooms}} + 68x_{\text{Apples}} + 11x_{\text{Carrots}} + 91x_{\text{Basil}} + 31x_{\text{Potatoes}} + 90x_{\text{Green Beans}} + 56x_{\text{Blueberries}} + 10x_{\text{Oranges}} + 24x_{\text{Watermelons}}
\]

**Subject to:**

\[
230x_{\text{Spinach}} + 637x_{\text{Shiitake Mushrooms}} + 773x_{\text{Apples}} + 653x_{\text{Carrots}} + 755x_{\text{Basil}} + 670x_{\text{Potatoes}} + 505x_{\text{Green Beans}} + 821x_{\text{Blueberries}} + 83x_{\text{Oranges}} + 249x_{\text{Watermelons}} \leq 875
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

**Where:**

- $x_{\text{Spinach}}$: units of Spinach to order
- $x_{\text{Shiitake Mushrooms}}$: units of Shiitake Mushrooms to order
- $x_{\text{Apples}}$: units of Apples to order
- $x_{\text{Carrots}}$: units of Carrots to order
- $x_{\text{Basil}}$: units of Basil to order
- $x_{\text{Potatoes}}$: units of Potatoes to order
- $x_{\text{Green Beans}}$: units of Green Beans to order
- $x_{\text{Blueberries}}$: units of Blueberries to order
- $x_{\text{Oranges}}$: units of Oranges to order
- $x_{\text{Watermelons}}$: units of Watermelons to order

All variables are nonnegative integers. The objective maximizes total value, and the constraint ensures the total weight does not exceed the daily capacity of 875.