Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the areas listed by ProductName in products.csv.

**Parameters:**

- For each area $i$ (ProductName):
    - $v_i$: Value (development benefit per unit) from products.csv
    - $w_i$: Weight (resource consumption per unit) from products.csv

- $C$: Capacity (overall development capacity) from capacity.csv

**Data (in source order):**

| ProductName         | Value ($v_i$) | Weight ($w_i$) |
|---------------------|--------------|---------------|
| Queens              | 469          | 954           |
| Brooklyn            | 290          | 650           |
| Manhattan           | 236          | 961           |
| Bronx               | 235          | 950           |
| Staten Island       | 745          | 379           |
| Harlem              | 684          | 776           |
| Upper East Side     | 444          | 381           |
| Lower Manhattan     | 172          | 808           |
| Midtown             | 1000         | 937           |
| Long Island City    | 336          | 608           |
| Williamsburg        | 546          | 912           |
| Bushwick            | 535          | 391           |
| Flatbush            | 539          | 465           |
| Greenpoint          | 831          | 490           |
| Park Slope          | 139          | 918           |
| Astoria             | 432          | 787           |
| Jackson Heights     | 627          | 347           |
| Flushing            | 629          | 274           |
| Sunnyside           | 292          | 642           |
| Ditmars             | 978          | 130           |

Overall development capacity: $C = 586$

---

### Mathematical Model

**Decision Variables:**

$$
x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in \{\text{areas listed above in source order}\}
$$

**Objective:**

$$
\max \sum_{i} v_i x_i
$$

**Subject to:**

$$
\sum_{i} w_i x_i \leq 586
$$

$$
x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i
$$

**Where:**

- $v_i$ and $w_i$ are as listed above for each area $i$ (ProductName, in source order).
- $C = 586$ is the overall development capacity.

**All data and identifiers are preserved in original file and row order.**