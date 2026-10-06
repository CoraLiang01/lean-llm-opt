Let $x_i$ be the number of units of bread type $i$ to order each day, where $i$ indexes the bread types as given by item_name in products.csv.

Objective:
$$
\max \; 888x_{\text{Baguette}} + 134x_{\text{Croissant}} + 129x_{\text{Sourdough}} + 370x_{\text{Rye Bread}} + 921x_{\text{Brioche}} + 765x_{\text{Focaccia}} + 154x_{\text{Ciabatta}} + 837x_{\text{Pita}} + 584x_{\text{Bagel}} + 365x_{\text{English Muffin}}
$$

Subject to:

Storage capacity constraint:
$$
4x_{\text{Baguette}} + 2x_{\text{Croissant}} + 4x_{\text{Sourdough}} + 3x_{\text{Rye Bread}} + 2x_{\text{Brioche}} + 1x_{\text{Focaccia}} + 2x_{\text{Ciabatta}} + 1x_{\text{Pita}} + 3x_{\text{Bagel}} + 3x_{\text{English Muffin}} \leq 180
$$

Integrality and nonnegativity:
$$
x_{\text{Baguette}},\; x_{\text{Croissant}},\; x_{\text{Sourdough}},\; x_{\text{Rye Bread}},\; x_{\text{Brioche}},\; x_{\text{Focaccia}},\; x_{\text{Ciabatta}},\; x_{\text{Pita}},\; x_{\text{Bagel}},\; x_{\text{English Muffin}} \in \mathbb{Z}_{\geq 0}
$$

Where:

- $x_i$ = number of units of bread type $i$ to order each day (integer, $\geq 0$)
- item_value = expected profit per unit (from products.csv)
- resource_requirement = storage space required per unit (from products.csv)
- resource_capacity = total available storage space per day (from capacity.csv, 180)

All coefficients and identifiers are as retrieved and in source order.