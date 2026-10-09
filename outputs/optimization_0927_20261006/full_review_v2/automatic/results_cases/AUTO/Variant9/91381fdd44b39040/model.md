Let $y_p$ be the integer number of standard rolls cut using pattern $p$.

Minimize the total number of standard rolls used:
$$
\min \; y_{P1} + y_{P2} + y_{P3} + y_{P4} + y_{P5} + y_{P6} + y_{P7} + y_{P8} + y_{P9}
$$

Subject to demand satisfaction for each item type:

For item A:
$$
4y_{P1} + 0y_{P2} + 0y_{P3} + 0y_{P4} + 2y_{P5} + 1y_{P6} + 0y_{P7} + 1y_{P8} + 2y_{P9} \geq 24
$$

For item B:
$$
0y_{P1} + 3y_{P2} + 0y_{P3} + 0y_{P4} + 1y_{P5} + 0y_{P6} + 1y_{P7} + 1y_{P8} + 0y_{P9} \geq 18
$$

For item C:
$$
0y_{P1} + 0y_{P2} + 2y_{P3} + 0y_{P4} + 0y_{P5} + 1y_{P6} + 0y_{P7} + 1y_{P8} + 0y_{P9} \geq 12
$$

For item D:
$$
0y_{P1} + 0y_{P2} + 0y_{P3} + 2y_{P4} + 0y_{P5} + 0y_{P6} + 1y_{P7} + 0y_{P8} + 1y_{P9} \geq 10
$$

Nonnegativity and integrality:
$$
y_{P1}, y_{P2}, y_{P3}, y_{P4}, y_{P5}, y_{P6}, y_{P7}, y_{P8}, y_{P9} \in \mathbb{Z}_{\geq 0}
$$

Where:

- Patterns and their yields (units per standard roll):

| Pattern | A | B | C | D |
|---------|---|---|---|---|
| P1      | 4 | 0 | 0 | 0 |
| P2      | 0 | 3 | 0 | 0 |
| P3      | 0 | 0 | 2 | 0 |
| P4      | 0 | 0 | 0 | 2 |
| P5      | 2 | 1 | 0 | 0 |
| P6      | 1 | 0 | 1 | 0 |
| P7      | 0 | 1 | 0 | 1 |
| P8      | 1 | 1 | 1 | 0 |
| P9      | 2 | 0 | 0 | 1 |

- Demands:

| Item | Demand |
|------|--------|
| A    | 24     |
| B    | 18     |
| C    | 12     |
| D    | 10     |