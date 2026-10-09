Let the set of areas be indexed by i, corresponding to the following ProductName values in the order given:

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

Let x_i = scale of development per day in area i (nonnegative integer, for i = 1,...,20).

Parameters (from products.csv, in order):

| i  | ProductName         | Value | Weight |
|----|---------------------|-------|--------|
| 1  | Queens              | 469   | 954    |
| 2  | Brooklyn            | 290   | 650    |
| 3  | Manhattan           | 236   | 961    |
| 4  | Bronx               | 235   | 950    |
| 5  | Staten Island       | 745   | 379    |
| 6  | Harlem              | 684   | 776    |
| 7  | Upper East Side     | 444   | 381    |
| 8  | Lower Manhattan     | 172   | 808    |
| 9  | Midtown             | 1000  | 937    |
| 10 | Long Island City    | 336   | 608    |
| 11 | Williamsburg        | 546   | 912    |
| 12 | Bushwick            | 535   | 391    |
| 13 | Flatbush            | 539   | 465    |
| 14 | Greenpoint          | 831   | 490    |
| 15 | Park Slope          | 139   | 918    |
| 16 | Astoria             | 432   | 787    |
| 17 | Jackson Heights     | 627   | 347    |
| 18 | Flushing            | 629   | 274    |
| 19 | Sunnyside           | 292   | 642    |
| 20 | Ditmars             | 978   | 130    |

Overall development capacity (from capacity.csv): 586

Mathematical Optimization Model:

Decision variables:
 x_i ∈ {0, 1, 2, ...} for i = 1,...,20

Objective:
 Maximize 469 x₁ + 290 x₂ + 236 x₃ + 235 x₄ + 745 x₅ + 684 x₆ + 444 x₇ + 172 x₈ + 1000 x₉ + 336 x₁₀ + 546 x₁₁ + 535 x₁₂ + 539 x₁₃ + 831 x₁₄ + 139 x₁₅ + 432 x₁₆ + 627 x₁₇ + 629 x₁₈ + 292 x₁₉ + 978 x₂₀

Subject to:
 954 x₁ + 650 x₂ + 961 x₃ + 950 x₄ + 379 x₅ + 776 x₆ + 381 x₇ + 808 x₈ + 937 x₉ + 608 x₁₀ + 912 x₁₁ + 391 x₁₂ + 465 x₁₃ + 490 x₁₄ + 918 x₁₅ + 787 x₁₆ + 347 x₁₇ + 274 x₁₈ + 642 x₁₉ + 130 x₂₀ ≤ 586

 x_i ≥ 0 and integer for all i = 1,...,20

Where:
 x₁ = scale of development per day in Queens
 x₂ = scale of development per day in Brooklyn
 ...
 x₂₀ = scale of development per day in Ditmars

This model maximizes the total development benefit while ensuring the total development resource consumption does not exceed the overall capacity of 586.