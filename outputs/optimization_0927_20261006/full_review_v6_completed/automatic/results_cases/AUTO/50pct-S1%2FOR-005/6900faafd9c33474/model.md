Let $x_i$ be the number of units of bread type $i$ to order each day, where $i$ indexes the following bread types (in source order):

- Baguette
- Croissant
- Sourdough
- Rye Bread
- Brioche
- Focaccia
- Ciabatta
- Pita
- Bagel
- English Muffin

The expected profit per unit and storage space required per unit for each bread type are as follows:

| item_name        | item_value | resource_requirement |
|------------------|-----------|---------------------|
| Baguette         | 888       | 4                   |
| Croissant        | 134       | 2                   |
| Sourdough        | 129       | 4                   |
| Rye Bread        | 370       | 3                   |
| Brioche          | 921       | 2                   |
| Focaccia         | 765       | 1                   |
| Ciabatta         | 154       | 2                   |
| Pita             | 837       | 1                   |
| Bagel            | 584       | 3                   |
| English Muffin   | 365       | 3                   |

The total storage capacity available each day is 180 units.

The mathematical model is:

Maximize total expected profit:
$$
\max \; 888x_{\text{Baguette}} + 134x_{\text{Croissant}} + 129x_{\text{Sourdough}} + 370x_{\text{Rye Bread}} + 921x_{\text{Brioche}} + 765x_{\text{Focaccia}} + 154x_{\text{Ciabatta}} + 837x_{\text{Pita}} + 584x_{\text{Bagel}} + 365x_{\text{English Muffin}}
$$

Subject to the storage capacity constraint:
$$
4x_{\text{Baguette}} + 2x_{\text{Croissant}} + 4x_{\text{Sourdough}} + 3x_{\text{Rye Bread}} + 2x_{\text{Brioche}} + 1x_{\text{Focaccia}} + 2x_{\text{Ciabatta}} + 1x_{\text{Pita}} + 3x_{\text{Bagel}} + 3x_{\text{English Muffin}} \leq 180
$$

And integer, nonnegativity constraints:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for all bread types } i
$$

Where the bread types $i$ are as listed above, and all coefficients are as given in the table.