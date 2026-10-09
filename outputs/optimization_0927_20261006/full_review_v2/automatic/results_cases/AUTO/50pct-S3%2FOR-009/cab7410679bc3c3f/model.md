Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the following areas in the order given:

1. Queens
2. Brooklyn
3. Manhattan
4. Bronx
5. Staten Island
6. Harlem
7. Upper East Side
8. Lower Manhattan
9. Midtown
10. Long Island City
11. Williamsburg
12. Bushwick
13. Flatbush
14. Greenpoint
15. Astoria
16. Jackson Heights
17. Flushing
18. Sunnyside
19. Ditmars

Let $v_i$ be the development benefit ("Value") and $w_i$ be the development resource requirement ("Weight") for area $i$, as given below.

The overall development capacity is $586$.

#### Parameters (in source order):

| $i$ | ProductName         | $v_i$ (Value) | $w_i$ (Weight) |
|-----|---------------------|---------------|---------------|
| 1   | Queens              | 469           | 954           |
| 2   | Brooklyn            | 290           | 650           |
| 3   | Manhattan           | 236           | 961           |
| 4   | Bronx               | 235           | 950           |
| 5   | Staten Island       | 745           | 379           |
| 6   | Harlem              | 684           | 776           |
| 7   | Upper East Side     | 444           | 381           |
| 8   | Lower Manhattan     | 172           | 808           |
| 9   | Midtown             | 1000          | 937           |
| 10  | Long Island City    | 336           | 608           |
| 11  | Williamsburg        | 546           | 912           |
| 12  | Bushwick            | 535           | 391           |
| 13  | Flatbush            | 539           | 465           |
| 14  | Greenpoint          | 831           | 490           |
| 15  | Astoria             | 432           | 787           |
| 16  | Jackson Heights     | 627           | 347           |
| 17  | Flushing            | 629           | 274           |
| 18  | Sunnyside           | 292           | 642           |
| 19  | Ditmars             | 978           | 130           |

#### Mathematical Model

Objective:
$$
\max \sum_{i=1}^{19} v_i x_i
$$

Subject to:
$$
\sum_{i=1}^{19} w_i x_i \leq 586
$$

$$
x_i \geq 0, \quad \forall i = 1, \ldots, 19
$$

where all $x_i$ are continuous variables representing the scale of development per day in each area.

#### Data Used

- Capacity: $586$
- For each area $i$:
    - $v_i$ = Value (from products.csv)
    - $w_i$ = Weight (from products.csv)
    - Area names and order as above

All coefficients and identifiers are preserved in source order.