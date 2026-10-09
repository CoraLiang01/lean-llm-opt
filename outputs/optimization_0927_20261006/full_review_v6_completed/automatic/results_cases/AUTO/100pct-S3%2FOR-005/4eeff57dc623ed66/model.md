Let $x_i$ be the number of units of bread type $i$ to order each day, where $i$ indexes the following bread types:

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

Parameters:

- $p_i$: expected profit per unit of bread $i$ (from "item_value")
- $a_i$: storage space required per unit of bread $i$ (from "resource_requirement")
- $C$: total storage capacity (from "resource_capacity")

Numerical values:

| $i$ (item_name)      | $p_i$ (item_value) | $a_i$ (resource_requirement) |
|----------------------|--------------------|------------------------------|
| Baguette             | 888                | 4                            |
| Croissant            | 134                | 2                            |
| Sourdough            | 129                | 4                            |
| Rye Bread            | 370                | 3                            |
| Brioche              | 921                | 2                            |
| Focaccia             | 765                | 1                            |
| Ciabatta             | 154                | 2                            |
| Pita                 | 837                | 1                            |
| Bagel                | 584                | 3                            |
| English Muffin       | 365                | 3                            |

Total storage capacity: $C = 180$

Mathematical Model:

Maximize total expected profit:
$$
\max \; 888x_{\text{Baguette}} + 134x_{\text{Croissant}} + 129x_{\text{Sourdough}} + 370x_{\text{Rye Bread}} + 921x_{\text{Brioche}} + 765x_{\text{Focaccia}} + 154x_{\text{Ciabatta}} + 837x_{\text{Pita}} + 584x_{\text{Bagel}} + 365x_{\text{English Muffin}}
$$

Subject to the storage capacity constraint:
$$
4x_{\text{Baguette}} + 2x_{\text{Croissant}} + 4x_{\text{Sourdough}} + 3x_{\text{Rye Bread}} + 2x_{\text{Brioche}} + 1x_{\text{Focaccia}} + 2x_{\text{Ciabatta}} + 1x_{\text{Pita}} + 3x_{\text{Bagel}} + 3x_{\text{English Muffin}} \leq 180
$$

Integrality and non-negativity:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

Where $x_i$ is the number of units of bread type $i$ to order each day.