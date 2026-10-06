Let the set of areas be indexed by i, corresponding to the rows in products.csv, in the given order.

Decision variables:
For each area i (i = 1,...,20), let x_i = scale of development per day in area i (nonnegative integer).

Parameters (from products.csv, in order):
Let ProductName_i, Value_i, Weight_i be the area name, benefit per unit, and resource consumption per unit for area i, respectively.

Data (in order):

| i | ProductName         | Value_i | Weight_i |
|---|---------------------|---------|----------|
| 1 | Queens              | 469     | 954      |
| 2 | Brooklyn            | 290     | 650      |
| 3 | Manhattan           | 236     | 961      |
| 4 | Bronx               | 235     | 950      |
| 5 | Staten Island       | 745     | 379      |
| 6 | Harlem              | 684     | 776      |
| 7 | Upper East Side     | 444     | 381      |
| 8 | Lower Manhattan     | 172     | 808      |
| 9 | Midtown             | 1000    | 937      |
|10 | Long Island City    | 336     | 608      |
|11 | Williamsburg        | 546     | 912      |
|12 | Bushwick            | 535     | 391      |
|13 | Flatbush            | 539     | 465      |
|14 | Greenpoint          | 831     | 490      |
|15 | Park Slope          | 139     | 918      |
|16 | Astoria             | 432     | 787      |
|17 | Jackson Heights     | 627     | 347      |
|18 | Flushing            | 629     | 274      |
|19 | Sunnyside           | 292     | 642      |
|20 | Ditmars             | 978     | 130      |

Capacity: 586

Mathematical Model:

Variables:
  x_i ∈ {0, 1, 2, ...} for i = 1,...,20

Objective:
  Maximize ∑_{i=1}^{20} Value_i * x_i
  = 469 x_1 + 290 x_2 + 236 x_3 + 235 x_4 + 745 x_5 + 684 x_6 + 444 x_7 + 172 x_8 + 1000 x_9 + 336 x_10 + 546 x_11 + 535 x_12 + 539 x_13 + 831 x_14 + 139 x_15 + 432 x_16 + 627 x_17 + 629 x_18 + 292 x_19 + 978 x_20

Subject to:
  954 x_1 + 650 x_2 + 961 x_3 + 950 x_4 + 379 x_5 + 776 x_6 + 381 x_7 + 808 x_8 + 937 x_9 + 608 x_10 + 912 x_11 + 391 x_12 + 465 x_13 + 490 x_14 + 918 x_15 + 787 x_16 + 347 x_17 + 274 x_18 + 642 x_19 + 130 x_20 ≤ 586

  x_i ≥ 0 and integer for all i = 1,...,20

Where:
  x_i = scale of development per day in area i (for each area as listed above, in order).

This model selects the scale of development per day in each area to maximize total benefit, subject to the overall development capacity constraint.