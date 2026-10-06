Let $x_i$ be the number of units of bread type $i$ to order each day. The index $i$ corresponds to each row in products.csv, using the item_name as the identifier.

Objective:
$$
\max \; 888x_{\text{Baguette}} + 134x_{\text{Croissant}} + 129x_{\text{Sourdough}} + 370x_{\text{Rye Bread}} + 921x_{\text{Brioche}} + 765x_{\text{Focaccia}} + 154x_{\text{Ciabatta}} + 837x_{\text{Pita}} + 584x_{\text{Bagel}} + 365x_{\text{English Muffin}}
$$

Subject to (storage capacity constraint from capacity.csv, using resource_requirement from products.csv):

$$
4x_{\text{Baguette}} + 2x_{\text{Croissant}} + 4x_{\text{Sourdough}} + 3x_{\text{Rye Bread}} + 2x_{\text{Brioche}} + 1x_{\text{Focaccia}} + 2x_{\text{Ciabatta}} + 1x_{\text{Pita}} + 3x_{\text{Bagel}} + 3x_{\text{English Muffin}} \leq 180
$$

Integrality:
$$
x_{\text{Baguette}},\; x_{\text{Croissant}},\; x_{\text{Sourdough}},\; x_{\text{Rye Bread}},\; x_{\text{Brioche}},\; x_{\text{Focaccia}},\; x_{\text{Ciabatta}},\; x_{\text{Pita}},\; x_{\text{Bagel}},\; x_{\text{English Muffin}} \in \mathbb{Z}_{\geq 0}
$$

Where:
- $x_{\text{Baguette}}$ = number of Baguette to order
- $x_{\text{Croissant}}$ = number of Croissant to order
- $x_{\text{Sourdough}}$ = number of Sourdough to order
- $x_{\text{Rye Bread}}$ = number of Rye Bread to order
- $x_{\text{Brioche}}$ = number of Brioche to order
- $x_{\text{Focaccia}}$ = number of Focaccia to order
- $x_{\text{Ciabatta}}$ = number of Ciabatta to order
- $x_{\text{Pita}}$ = number of Pita to order
- $x_{\text{Bagel}}$ = number of Bagel to order
- $x_{\text{English Muffin}}$ = number of English Muffin to order

All coefficients and identifiers are taken directly from the provided data.