Let $x_i$ be the number of units of product $i$ to be fulfilled, for each product $i$ listed below.

**Parameters (from data):**

| Product Name   | Revenue ($r_i$) | Demand ($d_i$) | Initial Inventory ($s_i$) |
|----------------|-----------------|---------------|---------------------------|
| sku_I27        | 238             | 6             | 30                        |
| sku_I499       | 287             | 4             | 20                        |
| sku_I719       | 268             | 16            | 80                        |
| sku_T18        | 318             | 14            | 70                        |
| sku_T29        | 207             | 4             | 20                        |
| sku_T39        | 258             | 32            | 160                       |
| sku_T499       | 249             | 8             | 40                        |
| sku_T9         | 227             | 2             | 10                        |
| sku_3081       | 198             | 10            | 50                        |
| sku_339        | 254             | 8             | 40                        |
| sku_3799       | 246             | 18            | 90                        |
| sku_439        | 258             | 2             | 10                        |
| sku_539        | 268             | 4             | 20                        |
| sku_61399      | 278             | 8             | 40                        |
| sku_628        | 268             | 2             | 10                        |
| sku_708        | 298             | 198           | 990                       |
| sku_77         | 258             | 32            | 160                       |
| sku_79         | 315             | 18            | 90                        |
| sku_799        | 264             | 570           | 2870                      |
| sku_8499       | 238             | 6             | 30                        |
| sku_89         | 258             | 26            | 130                       |
| sku_897        | 268             | 6             | 30                        |
| sku_9699       | 288             | 33            | 170                       |
| sku_bobo       | 228             | 33            | 170                       |

**Decision Variables:**

For each product $i$ (as listed above):

$\quad x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**

Maximize total revenue from fulfilled demand:
$$
\max \sum_{i} r_i x_i
$$

**Constraints:**

1. **Demand fulfillment cannot exceed demand:**
   $$
   x_i \leq d_i \qquad \forall i
   $$

2. **Cannot fulfill more than available inventory:**
   $$
   x_i \leq s_i \qquad \forall i
   $$

3. **Nonnegativity and integrality:**
   $$
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i
   $$

**Where:**

- $r_i$ = Revenue for product $i$ (see table above)
- $d_i$ = Demand for product $i$ (see table above)
- $s_i$ = Initial Inventory for product $i$ (see table above)
- $x_i$ = Number of units of product $i$ to fulfill

**All data and identifiers are preserved in source order as required.**