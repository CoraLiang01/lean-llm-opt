Let x_Baguette, x_Croissant, x_Sourdough, x_RyeBread, x_Brioche, x_Focaccia, x_Ciabatta, x_Pita, x_Bagel, x_EnglishMuffin be the integer number of units to order of each bread type per day.

Maximize total expected profit:
maximize
    888 x_Baguette + 134 x_Croissant + 129 x_Sourdough + 370 x_RyeBread + 921 x_Brioche + 765 x_Focaccia + 154 x_Ciabatta + 837 x_Pita + 584 x_Bagel + 365 x_EnglishMuffin

subject to the storage capacity constraint:
    4 x_Baguette + 2 x_Croissant + 4 x_Sourdough + 3 x_RyeBread + 2 x_Brioche + 1 x_Focaccia + 2 x_Ciabatta + 1 x_Pita + 3 x_Bagel + 3 x_EnglishMuffin ≤ 180

and integrality and nonnegativity:
    x_Baguette, x_Croissant, x_Sourdough, x_RyeBread, x_Brioche, x_Focaccia, x_Ciabatta, x_Pita, x_Bagel, x_EnglishMuffin ∈ {0, 1, 2, ...}

All coefficients and constraints are taken directly from the supplied CSV data.