##### Decision Variables:

Let $x_i$ denote the number of units of product $i$ (where $i$ indexes the set of ‘ELE-S’ products) to fulfill.

##### Objective Function:

$\quad \max \sum_{i} r_i x_i$

where $r_i$ is the revenue per unit for product $i$.

##### Constraints:

For each product $i$:
- $0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\}$

That is,
- $x_i \leq \text{Initial Inventory}_i$
- $x_i \leq \text{Demand}_i$
- $x_i \geq 0$

##### Retrieved Information

{
  "products": [
    {
      "Product_Reference": "ELE-SMA-10000463",
      "Revenue": 4.0,
      "Demand": 295,
      "Initial Inventory": 2000.0
    },
    {
      "Product_Reference": "ELE-SMA-10000487",
      "Revenue": 14.0,
      "Demand": 1002,
      "Initial Inventory": 7000.0
    },
    {
      "Product_Reference": "ELE-SMA-10003333",
      "Revenue": 14.0,
      "Demand": 958,
      "Initial Inventory": 7000.0
    },
    {
      "Product_Reference": "ELE-SMA-10009012",
      "Revenue": 4.0,
      "Demand": 777,
      "Initial Inventory": 6000.0
    },
    {
      "Product_Reference": "ELE-SMA-10009999",
      "Revenue": 4.0,
      "Demand": 271,
      "Initial Inventory": 2000.0
    },
    {
      "Product_Reference": "ELE-SMA-10011234",
      "Revenue": 4.0,
      "Demand": 244,
      "Initial Inventory": 2000.0
    },
    {
      "Product_Reference": "ELE-SMA-10027456",
      "Revenue": 14.0,
      "Demand": 990,
      "Initial Inventory": 7000.0
    },
    {
      "Product_Reference": "ELE-SMA-10028567",
      "Revenue": 14.0,
      "Demand": 1000,
      "Initial Inventory": 7000.0
    }
  ]
}

##### Model with explicit variables:

Let the set of products be indexed as follows:
- $i=1$: ELE-SMA-10000463
- $i=2$: ELE-SMA-10000487
- $i=3$: ELE-SMA-10003333
- $i=4$: ELE-SMA-10009012
- $i=5$: ELE-SMA-10009999
- $i=6$: ELE-SMA-10011234
- $i=7$: ELE-SMA-10027456
- $i=8$: ELE-SMA-10028567

Let $x_1, x_2, ..., x_8$ be the decision variables.

Objective:
$$
\max \; 4.0x_1 + 14.0x_2 + 14.0x_3 + 4.0x_4 + 4.0x_5 + 4.0x_6 + 14.0x_7 + 14.0x_8
$$

Subject to:
\[
\begin{align*}
0 \leq x_1 &\leq \min\{2000.0, 295\} = 295 \\
0 \leq x_2 &\leq \min\{7000.0, 1002\} = 1002 \\
0 \leq x_3 &\leq \min\{7000.0, 958\} = 958 \\
0 \leq x_4 &\leq \min\{6000.0, 777\} = 777 \\
0 \leq x_5 &\leq \min\{2000.0, 271\} = 271 \\
0 \leq x_6 &\leq \min\{2000.0, 244\} = 244 \\
0 \leq x_7 &\leq \min\{7000.0, 990\} = 990 \\
0 \leq x_8 &\leq \min\{7000.0, 1000\} = 1000 \\
\end{align*}
\]

All $x_i$ are continuous and non-negative.

##### Parameters (as retrieved):

| Product_Reference      | Revenue | Demand | Initial Inventory |
|-----------------------|---------|--------|------------------|
| ELE-SMA-10000463      | 4.0     | 295    | 2000.0           |
| ELE-SMA-10000487      | 14.0    | 1002   | 7000.0           |
| ELE-SMA-10003333      | 14.0    | 958    | 7000.0           |
| ELE-SMA-10009012      | 4.0     | 777    | 6000.0           |
| ELE-SMA-10009999      | 4.0     | 271    | 2000.0           |
| ELE-SMA-10011234      | 4.0     | 244    | 2000.0           |
| ELE-SMA-10027456      | 14.0    | 990    | 7000.0           |
| ELE-SMA-10028567      | 14.0    | 1000   | 7000.0           |