Let $I$ be the set of bread types:
$$
I = \{\text{Baguette},\ \text{Croissant},\ \text{Sourdough},\ \text{Rye Bread},\ \text{Brioche},\ \text{Focaccia},\ \text{Ciabatta},\ \text{Pita},\ \text{Bagel},\ \text{English Muffin}\}
$$

Let $x_i$ be the integer number of units of bread type $i \in I$ to order each day.

Parameters (from source order):

- Storage capacity: $180$
- For each bread type $i$:

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

Decision variables:
$$
x_i \in \mathbb{Z}_{\geq 0},\quad \forall i \in I
$$

Objective:
$$
\max\ 888x_{\text{Baguette}} + 134x_{\text{Croissant}} + 129x_{\text{Sourdough}} + 370x_{\text{Rye Bread}} + 921x_{\text{Brioche}} + 765x_{\text{Focaccia}} + 154x_{\text{Ciabatta}} + 837x_{\text{Pita}} + 584x_{\text{Bagel}} + 365x_{\text{English Muffin}}
$$

Subject to:
$$
4x_{\text{Baguette}} + 2x_{\text{Croissant}} + 4x_{\text{Sourdough}} + 3x_{\text{Rye Bread}} + 2x_{\text{Brioche}} + 1x_{\text{Focaccia}} + 2x_{\text{Ciabatta}} + 1x_{\text{Pita}} + 3x_{\text{Bagel}} + 3x_{\text{English Muffin}} \leq 180
$$
$$
x_i \in \mathbb{Z}_{\geq 0},\quad \forall i \in I
$$