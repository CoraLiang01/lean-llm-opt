Let the set of areas be indexed by i, corresponding to the rows in products.csv, in the given order:

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

Parameters from products.csv:

| i  | ProductName        | Value | Weight |
|----|--------------------|-------|--------|
| 1  | Queens             | 443   | 104    |
| 2  | Brooklyn           | 522   | 368    |
| 3  | Manhattan          | 300   | 483    |
| 4  | Bronx              | 767   | 165    |
| 5  | Staten Island      | 300   | 105    |
| 6  | Harlem             | 309   | 123    |
| 7  | Upper East Side    | 598   | 131    |
| 8  | Lower Manhattan    | 460   | 341    |
| 9  | Midtown            | 318   | 258    |
| 10 | Long Island City   | 126   | 469    |
| 11 | Williamsburg       | 593   | 387    |
| 12 | Bushwick           | 871   | 425    |
| 13 | Flatbush           | 858   | 482    |
| 14 | Greenpoint         | 321   | 495    |
| 15 | Park Slope         | 275   | 305    |
| 16 | Astoria            | 700   | 377    |
| 17 | Jackson Heights    | 685   | 318    |
| 18 | Flushing           | 940   | 56     |
| 19 | Sunnyside          | 522   | 213    |
| 20 | Ditmars            | 763   | 472    |

Overall development capacity from capacity.csv: 4466

Mathematical Model:

Decision variables:
 x_i ∈ {0, 1, 2, ...} for i = 1,...,20

Objective:
 Maximize Z = 443 x₁ + 522 x₂ + 300 x₃ + 767 x₄ + 300 x₅ + 309 x₆ + 598 x₇ + 460 x₈ + 318 x₉ + 126 x₁₀ + 593 x₁₁ + 871 x₁₂ + 858 x₁₃ + 321 x₁₄ + 275 x₁₅ + 700 x₁₆ + 685 x₁₇ + 940 x₁₈ + 522 x₁₉ + 763 x₂₀

Subject to:
 104 x₁ + 368 x₂ + 483 x₃ + 165 x₄ + 105 x₅ + 123 x₆ + 131 x₇ + 341 x₈ + 258 x₉ + 469 x₁₀ + 387 x₁₁ + 425 x₁₂ + 482 x₁₃ + 495 x₁₄ + 305 x₁₅ + 377 x₁₆ + 318 x₁₇ + 56 x₁₈ + 213 x₁₉ + 472 x₂₀ ≤ 4466

 x_i ≥ 0 and integer for all i = 1,...,20

Where:
 x₁ = Queens, x₂ = Brooklyn, x₃ = Manhattan, x₄ = Bronx, x₅ = Staten Island, x₆ = Harlem, x₇ = Upper East Side, x₈ = Lower Manhattan, x₉ = Midtown, x₁₀ = Long Island City, x₁₁ = Williamsburg, x₁₂ = Bushwick, x₁₃ = Flatbush, x₁₄ = Greenpoint, x₁₅ = Park Slope, x₁₆ = Astoria, x₁₇ = Jackson Heights, x₁₈ = Flushing, x₁₉ = Sunnyside, x₂₀ = Ditmars.

This is a complete integer optimization model for the developer's problem, using all provided data.