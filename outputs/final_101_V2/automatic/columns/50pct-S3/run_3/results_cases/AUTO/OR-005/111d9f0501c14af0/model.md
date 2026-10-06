##### Sets and Indices
Let $I$ be the set of bread types, indexed by $i$.

##### Parameters (from products.csv and capacity.csv, in source order)
- item_name (bread type): 
  1. Baguette
  2. Croissant
  3. Sourdough
  4. Rye Bread
  5. Brioche
  6. Focaccia
  7. Ciabatta
  8. Pita
  9. Bagel
  10. English Muffin

- item_value (expected profit per unit):  
  - Baguette: 888  
  - Croissant: 134  
  - Sourdough: 129  
  - Rye Bread: 370  
  - Brioche: 921  
  - Focaccia: 765  
  - Ciabatta: 154  
  - Pita: 837  
  - Bagel: 584  
  - English Muffin: 365  

- resource_requirement (storage units per unit):  
  - Baguette: 4  
  - Croissant: 2  
  - Sourdough: 4  
  - Rye Bread: 3  
  - Brioche: 2  
  - Focaccia: 1  
  - Ciabatta: 2  
  - Pita: 1  
  - Bagel: 3  
  - English Muffin: 3  

- resource_capacity (from capacity.csv, last row): $180$

##### Decision Variables
- $x_i$: Number of units of bread type $i$ to order each day, $x_i \in \mathbb{Z}_{\geq 0}$

##### Mathematical Model

Objective:
$$
\max \quad 888\,x_{\text{Baguette}} + 134\,x_{\text{Croissant}} + 129\,x_{\text{Sourdough}} + 370\,x_{\text{Rye Bread}} + 921\,x_{\text{Brioche}} + 765\,x_{\text{Focaccia}} + 154\,x_{\text{Ciabatta}} + 837\,x_{\text{Pita}} + 584\,x_{\text{Bagel}} + 365\,x_{\text{English Muffin}}
$$

Subject to:

Storage capacity constraint:
$$
4\,x_{\text{Baguette}} + 2\,x_{\text{Croissant}} + 4\,x_{\text{Sourdough}} + 3\,x_{\text{Rye Bread}} + 2\,x_{\text{Brioche}} + 1\,x_{\text{Focaccia}} + 2\,x_{\text{Ciabatta}} + 1\,x_{\text{Pita}} + 3\,x_{\text{Bagel}} + 3\,x_{\text{English Muffin}} \leq 180
$$

Integrality and nonnegativity:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

##### Source Data Used (in original order)

products.csv:
| item_name        | item_value | resource_requirement |
|------------------|------------|---------------------|
| Baguette         | 888        | 4                   |
| Croissant        | 134        | 2                   |
| Sourdough        | 129        | 4                   |
| Rye Bread        | 370        | 3                   |
| Brioche          | 921        | 2                   |
| Focaccia         | 765        | 1                   |
| Ciabatta         | 154        | 2                   |
| Pita             | 837        | 1                   |
| Bagel            | 584        | 3                   |
| English Muffin   | 365        | 3                   |

capacity.csv:
| resource_capacity |
|-------------------|
| 180               |