Let the index i run over the bread types in the order given in products.csv:
1: Baguette
2: Croissant
3: Sourdough
4: Rye Bread
5: Brioche
6: Focaccia
7: Ciabatta
8: Pita
9: Bagel
10: English Muffin

Let x_i = number of units of bread type i to order each day (integer, x_i ≥ 0).

Parameters:
- Value_i: expected profit per unit of bread type i
- Weight_i: storage space required per unit of bread type i
- Capacity = 180 (from capacity.csv)

Data (from products.csv, in order):

| i  | ProductName      | Value_i | Weight_i |
|----|------------------|---------|----------|
| 1  | Baguette         | 888     | 4        |
| 2  | Croissant        | 134     | 2        |
| 3  | Sourdough        | 129     | 4        |
| 4  | Rye Bread        | 370     | 3        |
| 5  | Brioche          | 921     | 2        |
| 6  | Focaccia         | 765     | 1        |
| 7  | Ciabatta         | 154     | 2        |
| 8  | Pita             | 837     | 1        |
| 9  | Bagel            | 584     | 3        |
| 10 | English Muffin   | 365     | 3        |

Mathematical Model:

Decision variables:
x_i ∈ {0, 1, 2, ...} for i = 1,...,10

Objective:
Maximize total expected profit:
maximize
 888 x₁ + 134 x₂ + 129 x₃ + 370 x₄ + 921 x₅ + 765 x₆ + 154 x₇ + 837 x₈ + 584 x₉ + 365 x₁₀

Subject to storage capacity:
 4 x₁ + 2 x₂ + 4 x₃ + 3 x₄ + 2 x₅ + 1 x₆ + 2 x₇ + 1 x₈ + 3 x₉ + 3 x₁₀ ≤ 180

and integrality/nonnegativity:
 x_i ∈ {0, 1, 2, ...} for i = 1,...,10

Where:
x₁ = Baguette, x₂ = Croissant, x₃ = Sourdough, x₄ = Rye Bread, x₅ = Brioche, x₆ = Focaccia, x₇ = Ciabatta, x₈ = Pita, x₉ = Bagel, x₁₀ = English Muffin.