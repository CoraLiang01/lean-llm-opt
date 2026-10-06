Let $x_i$ be the number of units of bread type $i$ to order each day, where $i$ indexes the rows of products.csv in the given order.

Maximize total expected profit:
$$
\max \; 888x_{\text{Baguette}} + 134x_{\text{Croissant}} + 129x_{\text{Sourdough}} + 370x_{\text{Rye Bread}} + 921x_{\text{Brioche}} + 765x_{\text{Focaccia}} + 154x_{\text{Ciabatta}} + 837x_{\text{Pita}} + 584x_{\text{Bagel}} + 365x_{\text{English Muffin}}
$$

Subject to the storage capacity constraint:
$$
4x_{\text{Baguette}} + 2x_{\text{Croissant}} + 4x_{\text{Sourdough}} + 3x_{\text{Rye Bread}} + 2x_{\text{Brioche}} + 1x_{\text{Focaccia}} + 2x_{\text{Ciabatta}} + 1x_{\text{Pita}} + 3x_{\text{Bagel}} + 3x_{\text{English Muffin}} \leq 180
$$

Integer and nonnegativity constraints:
$$
x_{\text{Baguette}},\; x_{\text{Croissant}},\; x_{\text{Sourdough}},\; x_{\text{Rye Bread}},\; x_{\text{Brioche}},\; x_{\text{Focaccia}},\; x_{\text{Ciabatta}},\; x_{\text{Pita}},\; x_{\text{Bagel}},\; x_{\text{English Muffin}} \in \mathbb{Z}_{\geq 0}
$$

Where:
- item_name and coefficients are as in products.csv (source order):
    1. Baguette: item_value = 888, resource_requirement = 4
    2. Croissant: item_value = 134, resource_requirement = 2
    3. Sourdough: item_value = 129, resource_requirement = 4
    4. Rye Bread: item_value = 370, resource_requirement = 3
    5. Brioche: item_value = 921, resource_requirement = 2
    6. Focaccia: item_value = 765, resource_requirement = 1
    7. Ciabatta: item_value = 154, resource_requirement = 2
    8. Pita: item_value = 837, resource_requirement = 1
    9. Bagel: item_value = 584, resource_requirement = 3
    10. English Muffin: item_value = 365, resource_requirement = 3

- resource_capacity from capacity.csv: 180

All variables are nonnegative integers.