Let the set of areas (in original file order) be indexed by i, with names as follows:

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
15. Park Slope
16. Astoria
17. Jackson Heights
18. Flushing
19. Sunnyside
20. Ditmars

Let x_i = daily scale of development in area i (nonnegative integer).

Parameters (from products.csv, in file order):

| i  | Area Name           | Value (Benefit Coefficient) | Weight (Development Units per x_i) |
|----|---------------------|-----------------------------|-------------------------------------|
| 1  | Queens              | 443                         | 104                                 |
| 2  | Brooklyn            | 522                         | 368                                 |
| 3  | Manhattan           | 300                         | 483                                 |
| 4  | Bronx               | 767                         | 165                                 |
| 5  | Staten Island       | 300                         | 105                                 |
| 6  | Harlem              | 309                         | 123                                 |
| 7  | Upper East Side     | 598                         | 131                                 |
| 8  | Lower Manhattan     | 460                         | 341                                 |
| 9  | Midtown             | 318                         | 258                                 |
|10  | Long Island City    | 126                         | 469                                 |
|11  | Williamsburg        | 593                         | 387                                 |
|12  | Bushwick            | 871                         | 425                                 |
|13  | Flatbush            | 858                         | 482                                 |
|14  | Greenpoint          | 321                         | 495                                 |
|15  | Park Slope          | 275                         | 305                                 |
|16  | Astoria             | 700                         | 377                                 |
|17  | Jackson Heights     | 685                         | 318                                 |
|18  | Flushing            | 940                         | 56                                  |
|19  | Sunnyside           | 522                         | 213                                 |
|20  | Ditmars             | 763                         | 472                                 |

Overall development capacity (from capacity.csv): 4466

Mathematical Model:

Decision variables:
 x_i ∈ {0, 1, 2, ...} for i = 1,...,20

Objective:
 Maximize total benefit:
  maximize 443 x_1 + 522 x_2 + 300 x_3 + 767 x_4 + 300 x_5 + 309 x_6 + 598 x_7 + 460 x_8 + 318 x_9 + 126 x_{10} + 593 x_{11} + 871 x_{12} + 858 x_{13} + 321 x_{14} + 275 x_{15} + 700 x_{16} + 685 x_{17} + 940 x_{18} + 522 x_{19} + 763 x_{20}

Subject to:

 104 x_1 + 368 x_2 + 483 x_3 + 165 x_4 + 105 x_5 + 123 x_6 + 131 x_7 + 341 x_8 + 258 x_9 + 469 x_{10} + 387 x_{11} + 425 x_{12} + 482 x_{13} + 495 x_{14} + 305 x_{15} + 377 x_{16} + 318 x_{17} + 56 x_{18} + 213 x_{19} + 472 x_{20} ≤ 4466

 x_i ∈ {0, 1, 2, ...} for all i = 1,...,20

All coefficients and area names are as given in the original files and order. This is a pure integer linear program.