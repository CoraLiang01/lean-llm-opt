Here are all transportation costs per unit from each supplier (S1–S5) to each branch/customer (C1–C5) as provided in "transportation_costs.csv". The data is preserved in its original matrix orientation, with explicit facility and customer IDs:

**Source: "transportation_costs.csv"**

| Supplier (Row) | C1      | C2      | C3      | C4      | C5      |
|----------------|---------|---------|---------|---------|---------|
| S1             | 150.74  | 0.02    | 49.13   | 2080.15 | 426.4   |
| S2             | 233.05  | 97.73   | 49.84   | 1982.39 | 23.96   |
| S3             | 55.68   | 935.61  | 4.03    | 73.09   | 525.32  |
| S4             | 1483.82 | 1801.08 | 112.16  | 816.05  | 107.01  |
| S5             | 1119.47 | 884.31  | 0.08    | 1544.95 | 543.67  |

- Each row corresponds to a supplier (S1–S5).
- Each column corresponds to a branch/customer (C1–C5).
- The value at (S_i, C_j) is the transportation cost per unit from supplier S_i to branch C_j.
- Matrix shape and axis orientation are preserved as in the source.