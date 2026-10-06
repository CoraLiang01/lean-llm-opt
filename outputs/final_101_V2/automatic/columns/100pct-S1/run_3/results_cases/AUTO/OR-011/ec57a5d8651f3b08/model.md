Let $x_i$ be the number of units of product $i$ to order each day.

**Parameters:**

From products.csv (in source order):

| ProductName           | Weight | Value |
|-----------------------|--------|-------|
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

From capacity.csv:

- Total stock capacity: $875$

**Decision Variables:**

- $x_i \in \mathbb{Z}_{\geq 0}$, for each product $i$ (in the order above).

---

**Mathematical Model:**

**Objective:**
\[
\max \; 64x_{\text{Spinach}} + 75x_{\text{Shiitake Mushrooms}} + 68x_{\text{Apples}} + 11x_{\text{Carrots}} + 91x_{\text{Basil}} + 31x_{\text{Potatoes}} + 90x_{\text{Green Beans}} + 56x_{\text{Blueberries}} + 10x_{\text{Oranges}} + 24x_{\text{Watermelons}}
\]

**Subject to:**
\[
230x_{\text{Spinach}} + 637x_{\text{Shiitake Mushrooms}} + 773x_{\text{Apples}} + 653x_{\text{Carrots}} + 755x_{\text{Basil}} + 670x_{\text{Potatoes}} + 505x_{\text{Green Beans}} + 821x_{\text{Blueberries}} + 83x_{\text{Oranges}} + 249x_{\text{Watermelons}} \leq 875
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for all products } i
\]

**Where:**

- $x_{\text{Spinach}}$ = number of units of Spinach to order each day
- $x_{\text{Shiitake Mushrooms}}$ = number of units of Shiitake Mushrooms to order each day
- $x_{\text{Apples}}$ = number of units of Apples to order each day
- $x_{\text{Carrots}}$ = number of units of Carrots to order each day
- $x_{\text{Basil}}$ = number of units of Basil to order each day
- $x_{\text{Potatoes}}$ = number of units of Potatoes to order each day
- $x_{\text{Green Beans}}$ = number of units of Green Beans to order each day
- $x_{\text{Blueberries}}$ = number of units of Blueberries to order each day
- $x_{\text{Oranges}}$ = number of units of Oranges to order each day
- $x_{\text{Watermelons}}$ = number of units of Watermelons to order each day