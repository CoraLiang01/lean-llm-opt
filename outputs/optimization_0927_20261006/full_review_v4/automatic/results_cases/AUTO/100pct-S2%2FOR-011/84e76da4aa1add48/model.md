Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products listed below. Each $x_i$ is a nonnegative integer.

**Objective:**
\[
\max \; 64\,x_{\text{Spinach}} + 75\,x_{\text{Shiitake Mushrooms}} + 68\,x_{\text{Apples}} + 11\,x_{\text{Carrots}} + 91\,x_{\text{Basil}} + 31\,x_{\text{Potatoes}} + 90\,x_{\text{Green Beans}} + 56\,x_{\text{Blueberries}} + 10\,x_{\text{Oranges}} + 24\,x_{\text{Watermelons}}
\]

**Subject to:**

\[
230\,x_{\text{Spinach}}
+ 637\,x_{\text{Shiitake Mushrooms}}
+ 773\,x_{\text{Apples}}
+ 653\,x_{\text{Carrots}}
+ 755\,x_{\text{Basil}}
+ 670\,x_{\text{Potatoes}}
+ 505\,x_{\text{Green Beans}}
+ 821\,x_{\text{Blueberries}}
+ 83\,x_{\text{Oranges}}
+ 249\,x_{\text{Watermelons}}
\leq 875
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for all products } i
\]

**Where:**

- Products and their parameters (in source order):

| Product Name           | Value | Weight |
|------------------------|-------|--------|
| Spinach                | 64    | 230    |
| Shiitake Mushrooms     | 75    | 637    |
| Apples                 | 68    | 773    |
| Carrots                | 11    | 653    |
| Basil                  | 91    | 755    |
| Potatoes               | 31    | 670    |
| Green Beans            | 90    | 505    |
| Blueberries            | 56    | 821    |
| Oranges                | 10    | 83     |
| Watermelons            | 24    | 249    |

- The total stock capacity is 875.

**Decision variables:**
- $x_{\text{Spinach}}, x_{\text{Shiitake Mushrooms}}, x_{\text{Apples}}, x_{\text{Carrots}}, x_{\text{Basil}}, x_{\text{Potatoes}}, x_{\text{Green Beans}}, x_{\text{Blueberries}}, x_{\text{Oranges}}, x_{\text{Watermelons}} \in \mathbb{Z}_{\geq 0}$

**Maximize total value of ordered products, subject to the overall stock capacity.**