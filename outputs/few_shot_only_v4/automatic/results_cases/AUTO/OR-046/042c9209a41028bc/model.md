Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products as listed in the table below.

**Objective Function:**

$$
\max \; 64x_{\text{Spinach}} + 75x_{\text{Shiitake Mushrooms}} + 68x_{\text{Apples}} + 11x_{\text{Carrots}} + 91x_{\text{Basil}} + 31x_{\text{Potatoes}} + 90x_{\text{Green Beans}} + 56x_{\text{Blueberries}} + 10x_{\text{Oranges}} + 24x_{\text{Watermelons}}
$$

**Subject to:**

$$
230x_{\text{Spinach}} + 637x_{\text{Shiitake Mushrooms}} + 773x_{\text{Apples}} + 653x_{\text{Carrots}} + 755x_{\text{Basil}} + 670x_{\text{Potatoes}} + 505x_{\text{Green Beans}} + 821x_{\text{Blueberries}} + 83x_{\text{Oranges}} + 249x_{\text{Watermelons}} \leq 875
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{Spinach}, \text{Shiitake Mushrooms}, \text{Apples}, \text{Carrots}, \text{Basil}, \text{Potatoes}, \text{Green Beans}, \text{Blueberries}, \text{Oranges}, \text{Watermelons}\}
$$

**Where:**

| Product Name         | Weight | Value |
|---------------------|--------|-------|
| Spinach             | 230    | 64    |
| Shiitake Mushrooms  | 637    | 75    |
| Apples              | 773    | 68    |
| Carrots             | 653    | 11    |
| Basil               | 755    | 91    |
| Potatoes            | 670    | 31    |
| Green Beans         | 505    | 90    |
| Blueberries         | 821    | 56    |
| Oranges             | 83     | 10    |
| Watermelons         | 249    | 24    |

**Capacity:** $875$ (from capacity.csv)

**Decision variables:** $x_i$ are nonnegative integers representing the number of units of each product to order.