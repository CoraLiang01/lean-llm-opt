##### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$: Number of units of bread type $i$ to order each day, for each $i \in \mathcal{P}$ (where $\mathcal{P}$ is the set of all bread types).

##### Parameters

- $\mathcal{P} = \{$Baguette, Croissant, Sourdough, Rye Bread, Brioche, Focaccia, Ciabatta, Pita, Bagel, English Muffin$\}$
- Expected profit per unit ($v_i$) and storage weight per unit ($w_i$) for each bread type $i$:

| Product Name     | $v_i$ (Value) | $w_i$ (Weight) |
|------------------|:-------------:|:--------------:|
| Baguette         | 888           | 4              |
| Croissant        | 134           | 2              |
| Sourdough        | 129           | 4              |
| Rye Bread        | 370           | 3              |
| Brioche          | 921           | 2              |
| Focaccia         | 765           | 1              |
| Ciabatta         | 154           | 2              |
| Pita             | 837           | 1              |
| Bagel            | 584           | 3              |
| English Muffin   | 365           | 3              |

- Total storage capacity: $C = 180$

##### Objective Function

\[
\max \sum_{i \in \mathcal{P}} v_i x_i
\]
That is,
\[
\max \big(
888\,x_{\text{Baguette}} + 134\,x_{\text{Croissant}} + 129\,x_{\text{Sourdough}} + 370\,x_{\text{Rye Bread}} + 921\,x_{\text{Brioche}} + 765\,x_{\text{Focaccia}} + 154\,x_{\text{Ciabatta}} + 837\,x_{\text{Pita}} + 584\,x_{\text{Bagel}} + 365\,x_{\text{English Muffin}}
\big)
\]

##### Constraints

1. Storage capacity:
   \[
   4\,x_{\text{Baguette}} + 2\,x_{\text{Croissant}} + 4\,x_{\text{Sourdough}} + 3\,x_{\text{Rye Bread}} + 2\,x_{\text{Brioche}} + 1\,x_{\text{Focaccia}} + 2\,x_{\text{Ciabatta}} + 1\,x_{\text{Pita}} + 3\,x_{\text{Bagel}} + 3\,x_{\text{English Muffin}} \leq 180
   \]

2. Integer and nonnegativity:
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \mathcal{P}
   \]

##### Retrieved Information

- Capacity: $C = 180$
- Products, values, and weights:

  - Baguette: Value = 888, Weight = 4
  - Croissant: Value = 134, Weight = 2
  - Sourdough: Value = 129, Weight = 4
  - Rye Bread: Value = 370, Weight = 3
  - Brioche: Value = 921, Weight = 2
  - Focaccia: Value = 765, Weight = 1
  - Ciabatta: Value = 154, Weight = 2
  - Pita: Value = 837, Weight = 1
  - Bagel: Value = 584, Weight = 3
  - English Muffin: Value = 365, Weight = 3