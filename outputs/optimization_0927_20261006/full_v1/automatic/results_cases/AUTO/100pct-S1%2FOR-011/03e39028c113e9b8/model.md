Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products in the order they appear in products.csv.

**Objective:**
\[
\max \; 64x_{\text{Spinach}} + 75x_{\text{Shiitake Mushrooms}} + 68x_{\text{Apples}} + 11x_{\text{Carrots}} + 91x_{\text{Basil}} + 31x_{\text{Potatoes}} + 90x_{\text{Green Beans}} + 56x_{\text{Blueberries}} + 10x_{\text{Oranges}} + 24x_{\text{Watermelons}}
\]

**Subject to:**

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

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for all products } i
\]

**Where:**

- Products and their parameters (in source order):

| ProductName           | Value | Weight |
|-----------------------|-------|--------|
| Spinach               | 64    | 230    |
| Shiitake Mushrooms    | 75    | 637    |
| Apples                | 68    | 773    |
| Carrots               | 11    | 653    |
| Basil                 | 91    | 755    |
| Potatoes              | 31    | 670    |
| Green Beans           | 90    | 505    |
| Blueberries           | 56    | 821    |
| Oranges               | 10    | 83     |
| Watermelons           | 24    | 249    |

- The total stock capacity is 875.

**Decision variables:**
- $x_{\text{Spinach}}, x_{\text{Shiitake Mushrooms}}, x_{\text{Apples}}, x_{\text{Carrots}}, x_{\text{Basil}}, x_{\text{Potatoes}}, x_{\text{Green Beans}}, x_{\text{Blueberries}}, x_{\text{Oranges}}, x_{\text{Watermelons}}$ are nonnegative integers.