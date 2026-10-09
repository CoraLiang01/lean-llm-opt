##### Decision Variables:

Let $x_i$ denote the number of units of Organ product $i$ to be fulfilled, for each $i$ in the set of Organ products.

##### Objective Function:

$\quad \max \sum_{i} \text{Revenue}_i \cdot x_i$

##### Constraints:

For each Organ product $i$:
- $0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\}$

##### Retrieved Information

{
  "products": [
    {
      "Sub Category": "Organic Fruits",
      "Revenue": 60.8,
      "Demand": 678906,
      "Initial Inventory": 5034020.0
    },
    {
      "Sub Category": "Organic Staples",
      "Revenue": 918.45,
      "Demand": 749927,
      "Initial Inventory": 5589290.0
    },
    {
      "Sub Category": "Organic Vegetables",
      "Revenue": 77.52,
      "Demand": 699808,
      "Initial Inventory": 5202710.0
    }
  ]
}

##### Model in Detail

Let the set of Organ products be indexed by $i \in \{\text{Organic Fruits}, \text{Organic Staples}, \text{Organic Vegetables}\}$.

- $\text{Revenue}_i$:
  - Organic Fruits: $60.8$
  - Organic Staples: $918.45$
  - Organic Vegetables: $77.52$

- $\text{Initial Inventory}_i$:
  - Organic Fruits: $5,\!034,\!020.0$
  - Organic Staples: $5,\!589,\!290.0$
  - Organic Vegetables: $5,\!202,\!710.0$

- $\text{Demand}_i$:
  - Organic Fruits: $678,\!906$
  - Organic Staples: $749,\!927$
  - Organic Vegetables: $699,\!808$

The model:

$\max \left(60.8\, x_{\text{Organic Fruits}} + 918.45\, x_{\text{Organic Staples}} + 77.52\, x_{\text{Organic Vegetables}}\right)$

Subject to:

$0 \leq x_{\text{Organic Fruits}} \leq \min\{5,\!034,\!020.0,\ 678,\!906\}$

$0 \leq x_{\text{Organic Staples}} \leq \min\{5,\!589,\!290.0,\ 749,\!927\}$

$0 \leq x_{\text{Organic Vegetables}} \leq \min\{5,\!202,\!710.0,\ 699,\!808\}$

$x_i \geq 0$ and integer (if only whole units can be fulfilled), for all $i$.