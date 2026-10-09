Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products as listed below.

**Parameters:**

| ProductName           | Weight | Value |
|---------------------- |--------|-------|
| Spinach               | 230    | 64    |
| Shiitake Mushrooms    | 637    | 75    |
| Apples                | 773    | 68    |
| Carrots               | 653    | 11    |
| Basil                 | 755    | 91    |
| Potatoes              | 670    | 31    |
| Green Beans           | 505    | 90    |
| Blueberries           | 821    | 56    |
| Oranges               | 83     | 10    |
| Watermelons           | 249    | 24    |

Total stock capacity: $875$

---

**Mathematical Model:**

**Objective:**
\[
\max \; 64\,x_{\text{Spinach}} + 75\,x_{\text{Shiitake Mushrooms}} + 68\,x_{\text{Apples}} + 11\,x_{\text{Carrots}} + 91\,x_{\text{Basil}} + 31\,x_{\text{Potatoes}} + 90\,x_{\text{Green Beans}} + 56\,x_{\text{Blueberries}} + 10\,x_{\text{Oranges}} + 24\,x_{\text{Watermelons}}
\]

**Subject to:**
\[
230\,x_{\text{Spinach}} + 637\,x_{\text{Shiitake Mushrooms}} + 773\,x_{\text{Apples}} + 653\,x_{\text{Carrots}} + 755\,x_{\text{Basil}} + 670\,x_{\text{Potatoes}} + 505\,x_{\text{Green Beans}} + 821\,x_{\text{Blueberries}} + 83\,x_{\text{Oranges}} + 249\,x_{\text{Watermelons}} \leq 875
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for all products } i
\]

Where each $x_i$ is the number of units of product $i$ to order each day.