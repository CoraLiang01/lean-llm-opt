Let the set of areas (indexed by i) be as follows, in the order from products.csv:
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

Let x_i = daily scale of development in area i (integer, x_i ≥ 0).

Parameters (from products.csv, in order):
- Benefit coefficients (Value): [443, 522, 300, 767, 300, 309, 598, 460, 318, 126, 593, 871, 858, 321, 275, 700, 685, 940, 522, 763]
- Overall development capacity (from capacity.csv): 4466

The integer optimization model is:

Variables:
For i = 1,...,20,
 x_i ∈ {0, 1, 2, ...} (integer, nonnegative)

Objective:
Maximize total benefit:
 Maximize 443·x₁ + 522·x₂ + 300·x₃ + 767·x₄ + 300·x₅ + 309·x₆ + 598·x₇ + 460·x₈ + 318·x₉ + 126·x₁₀ + 593·x₁₁ + 871·x₁₂ + 858·x₁₃ + 321·x₁₄ + 275·x₁₅ + 700·x₁₆ + 685·x₁₇ + 940·x₁₈ + 522·x₁₉ + 763·x₂₀

Subject to:
 x₁ + x₂ + x₃ + x₄ + x₅ + x₆ + x₇ + x₈ + x₉ + x₁₀ + x₁₁ + x₁₂ + x₁₃ + x₁₄ + x₁₅ + x₁₆ + x₁₇ + x₁₈ + x₁₉ + x₂₀ ≤ 4466

 x_i ∈ {0, 1, 2, ...} for all i = 1,...,20

Where the mapping of i to area is as listed above.