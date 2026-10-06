Let the set of areas be indexed by i, corresponding to the rows in products.csv, with ProductName_i, Value_i, and Weight_i as given.

Decision variables:
For each area i (in the order listed in products.csv), let x_i = number of real estate units to develop per day in area i (x_i ∈ {0, 1, 2, ...}).

Parameters (from products.csv and capacity.csv):

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

Total development capacity: 586

Mathematical Model:

Variables:
 x_i ∈ {0, 1, 2, ...} for i = 1,...,20

Objective:
 Maximize Z = 469 x_1 + 290 x_2 + 236 x_3 + 235 x_4 + 745 x_5 + 684 x_6 + 444 x_7 + 172 x_8 + 1000 x_9 + 336 x_{10} + 546 x_{11} + 535 x_{12} + 539 x_{13} + 831 x_{14} + 139 x_{15} + 432 x_{16} + 627 x_{17} + 629 x_{18} + 292 x_{19} + 978 x_{20}

Subject to:
 954 x_1 + 650 x_2 + 961 x_3 + 950 x_4 + 379 x_5 + 776 x_6 + 381 x_7 + 808 x_8 + 937 x_9 + 608 x_{10} + 912 x_{11} + 391 x_{12} + 465 x_{13} + 490 x_{14} + 918 x_{15} + 787 x_{16} + 347 x_{17} + 274 x_{18} + 642 x_{19} + 130 x_{20} ≤ 586

 x_i ∈ {0, 1, 2, ...} for i = 1,...,20

Where the mapping of i to ProductName is as listed above.

This model maximizes the total development benefit while ensuring the total development scale does not exceed the available capacity.