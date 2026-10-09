Let $x_i$ be the number of units of bread type $i$ to order each day. Each $x_i$ is a nonnegative integer.

Let the set of bread types and their parameters be as follows (in source order):

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

The total storage capacity is 180.

The mathematical model is:

Objective:
$$
\max\ 888x_{\text{Baguette}} + 134x_{\text{Croissant}} + 129x_{\text{Sourdough}} + 370x_{\text{Rye Bread}} + 921x_{\text{Brioche}} + 765x_{\text{Focaccia}} + 154x_{\text{Ciabatta}} + 837x_{\text{Pita}} + 584x_{\text{Bagel}} + 365x_{\text{English Muffin}}
$$

Subject to:
$$
4x_{\text{Baguette}} + 2x_{\text{Croissant}} + 4x_{\text{Sourdough}} + 3x_{\text{Rye Bread}} + 2x_{\text{Brioche}} + 1x_{\text{Focaccia}} + 2x_{\text{Ciabatta}} + 1x_{\text{Pita}} + 3x_{\text{Bagel}} + 3x_{\text{English Muffin}} \leq 180
$$

$$
x_{\text{Baguette}},\ x_{\text{Croissant}},\ x_{\text{Sourdough}},\ x_{\text{Rye Bread}},\ x_{\text{Brioche}},\ x_{\text{Focaccia}},\ x_{\text{Ciabatta}},\ x_{\text{Pita}},\ x_{\text{Bagel}},\ x_{\text{English Muffin}} \in \mathbb{Z}_{\geq 0}
$$

Where:
- $x_{\text{Baguette}}$ = number of Baguette units to order per day
- $x_{\text{Croissant}}$ = number of Croissant units to order per day
- $x_{\text{Sourdough}}$ = number of Sourdough units to order per day
- $x_{\text{Rye Bread}}$ = number of Rye Bread units to order per day
- $x_{\text{Brioche}}$ = number of Brioche units to order per day
- $x_{\text{Focaccia}}$ = number of Focaccia units to order per day
- $x_{\text{Ciabatta}}$ = number of Ciabatta units to order per day
- $x_{\text{Pita}}$ = number of Pita units to order per day
- $x_{\text{Bagel}}$ = number of Bagel units to order per day
- $x_{\text{English Muffin}}$ = number of English Muffin units to order per day

All variables are nonnegative integers.