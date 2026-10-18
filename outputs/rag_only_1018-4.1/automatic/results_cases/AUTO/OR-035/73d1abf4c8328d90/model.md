Let the decision variables be:
x_Baguette: number of Baguette units to order each day (integer, ≥ 0)
x_Croissant: number of Croissant units to order each day (integer, ≥ 0)
x_Sourdough: number of Sourdough units to order each day (integer, ≥ 0)
x_RyeBread: number of Rye Bread units to order each day (integer, ≥ 0)
x_Brioche: number of Brioche units to order each day (integer, ≥ 0)
x_Focaccia: number of Focaccia units to order each day (integer, ≥ 0)
x_Ciabatta: number of Ciabatta units to order each day (integer, ≥ 0)
x_Pita: number of Pita units to order each day (integer, ≥ 0)
x_Bagel: number of Bagel units to order each day (integer, ≥ 0)
x_EnglishMuffin: number of English Muffin units to order each day (integer, ≥ 0)

Parameters from products.csv:
- Expected profit per unit (Value) and storage weight per unit (Weight):

| ProductName      | Value | Weight |
|------------------|-------|--------|
| Baguette         | 888   | 4      |
| Croissant        | 134   | 2      |
| Sourdough        | 129   | 4      |
| Rye Bread        | 370   | 3      |
| Brioche          | 921   | 2      |
| Focaccia         | 765   | 1      |
| Ciabatta         | 154   | 2      |
| Pita             | 837   | 1      |
| Bagel            | 584   | 3      |
| English Muffin   | 365   | 3      |

From capacity.csv:
- Total storage capacity per day: 180

Mathematical Model:

Variables:
x_Baguette, x_Croissant, x_Sourdough, x_RyeBread, x_Brioche, x_Focaccia, x_Ciabatta, x_Pita, x_Bagel, x_EnglishMuffin ∈ {0, 1, 2, ...}

Objective:
Maximize total expected profit:
maximize
  888 x_Baguette
+ 134 x_Croissant
+ 129 x_Sourdough
+ 370 x_RyeBread
+ 921 x_Brioche
+ 765 x_Focaccia
+ 154 x_Ciabatta
+ 837 x_Pita
+ 584 x_Bagel
+ 365 x_EnglishMuffin

Subject to storage capacity:
4 x_Baguette
+ 2 x_Croissant
+ 4 x_Sourdough
+ 3 x_RyeBread
+ 2 x_Brioche
+ 1 x_Focaccia
+ 2 x_Ciabatta
+ 1 x_Pita
+ 3 x_Bagel
+ 3 x_EnglishMuffin
≤ 180

And nonnegativity and integrality:
x_Baguette ≥ 0 and integer
x_Croissant ≥ 0 and integer
x_Sourdough ≥ 0 and integer
x_RyeBread ≥ 0 and integer
x_Brioche ≥ 0 and integer
x_Focaccia ≥ 0 and integer
x_Ciabatta ≥ 0 and integer
x_Pita ≥ 0 and integer
x_Bagel ≥ 0 and integer
x_EnglishMuffin ≥ 0 and integer

This model uses all provided data, aligns profit and storage coefficients by product, and enforces the single storage capacity constraint as specified.