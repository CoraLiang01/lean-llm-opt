Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products listed below. All $x_i$ are nonnegative integers.

**Parameters (from products.csv and capacity.csv):**

| ProductName           | Value | Weight |
|---------------------- |-------|--------|
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

Total stock capacity: $875$

---

**Mathematical Model**

**Decision variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$, for each product $i$ in the table above.

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

Where:
- $x_{\text{Spinach}}$ = units of Spinach to order
- $x_{\text{Shiitake Mushrooms}}$ = units of Shiitake Mushrooms to order
- $x_{\text{Apples}}$ = units of Apples to order
- $x_{\text{Carrots}}$ = units of Carrots to order
- $x_{\text{Basil}}$ = units of Basil to order
- $x_{\text{Potatoes}}$ = units of Potatoes to order
- $x_{\text{Green Beans}}$ = units of Green Beans to order
- $x_{\text{Blueberries}}$ = units of Blueberries to order
- $x_{\text{Oranges}}$ = units of Oranges to order
- $x_{\text{Watermelons}}$ = units of Watermelons to order

**All variables are nonnegative integers.**