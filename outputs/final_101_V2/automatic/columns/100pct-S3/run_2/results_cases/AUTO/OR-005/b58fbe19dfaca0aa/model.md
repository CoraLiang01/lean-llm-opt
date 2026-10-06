Let $x_i$ be the number of units of bread type $i$ to order each day. The variables $x_i$ are nonnegative integers.

Let $I$ be the set of bread types, indexed by item_name as given in products.csv.

Let $p_i$ be the expected profit per unit of bread type $i$ (item_value from products.csv).

Let $a_i$ be the resource requirement per unit of bread type $i$ (resource_requirement from products.csv).

Let $C$ be the storage capacity (resource_capacity from capacity.csv).

The complete mathematical model is:

Objective:
$$
\max \sum_{i \in I} p_i x_i
$$

Subject to:
$$
\sum_{i \in I} a_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

Where:

- $I = \{$Baguette, Croissant, Sourdough, Rye Bread, Brioche, Focaccia, Ciabatta, Pita, Bagel, English Muffin$\}$
- $p_i$ (item_value): 
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
- $a_i$ (resource_requirement): 
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
- $C$ (resource_capacity): 180

All data is used as retrieved and in original order.