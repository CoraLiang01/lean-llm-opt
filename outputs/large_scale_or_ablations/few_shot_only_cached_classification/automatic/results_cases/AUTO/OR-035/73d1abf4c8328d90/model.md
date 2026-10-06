##### Sets and Indices
Let $i$ index the bread types, with identifiers as in the ProductName column.

##### Parameters (from products.csv and capacity.csv)
- $p_i$: Expected profit per unit of bread type $i$ (Value column)
- $w_i$: Storage space required per unit of bread type $i$ (Weight column)
- $C$: Total storage capacity (Capacity column)

##### Decision Variables
- $x_i$: Number of units of bread type $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

##### Data

| $i$                | $p_i$ | $w_i$ |
|--------------------|-------|-------|
| Baguette           | 888   | 4     |
| Croissant          | 134   | 2     |
| Sourdough          | 129   | 4     |
| Rye Bread          | 370   | 3     |
| Brioche            | 921   | 2     |
| Focaccia           | 765   | 1     |
| Ciabatta           | 154   | 2     |
| Pita               | 837   | 1     |
| Bagel              | 584   | 3     |
| English Muffin     | 365   | 3     |

$C = 180$

##### Mathematical Model

Objective:
$$
\max \; 888x_{\text{Baguette}} + 134x_{\text{Croissant}} + 129x_{\text{Sourdough}} + 370x_{\text{Rye Bread}} + 921x_{\text{Brioche}} + 765x_{\text{Focaccia}} + 154x_{\text{Ciabatta}} + 837x_{\text{Pita}} + 584x_{\text{Bagel}} + 365x_{\text{English Muffin}}
$$

Subject to:
$$
4x_{\text{Baguette}} + 2x_{\text{Croissant}} + 4x_{\text{Sourdough}} + 3x_{\text{Rye Bread}} + 2x_{\text{Brioche}} + 1x_{\text{Focaccia}} + 2x_{\text{Ciabatta}} + 1x_{\text{Pita}} + 3x_{\text{Bagel}} + 3x_{\text{English Muffin}} \leq 180
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{Baguette}, \text{Croissant}, \text{Sourdough}, \text{Rye Bread}, \text{Brioche}, \text{Focaccia}, \text{Ciabatta}, \text{Pita}, \text{Bagel}, \text{English Muffin}\}
$$