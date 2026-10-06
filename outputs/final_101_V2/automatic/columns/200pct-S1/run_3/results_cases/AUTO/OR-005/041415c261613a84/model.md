Let $x_i$ be the number of units of bread type $i$ to order each day, where $i$ indexes the bread types as given by item_name in products.csv.

Parameters (from products.csv):

- For each bread type $i$:
    - item_name: name of bread type
    - item_value: expected profit per unit of bread type $i$
    - resource_requirement: storage space required per unit of bread type $i$

Parameter (from capacity.csv):

- resource_capacity: total available storage space per day (180 units)

Indices:

- Let $I$ be the set of bread types as given by the item_name column in products.csv.

Model:

Objective:
$$
\max \sum_{i \in I} \text{item\_value}_i \cdot x_i
$$

Subject to:

Storage capacity constraint:
$$
\sum_{i \in I} \text{resource\_requirement}_i \cdot x_i \leq 180
$$

Integrality and nonnegativity:
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
$$

Where:

- $I = \{$Baguette, Croissant, Sourdough, Rye Bread, Brioche, Focaccia, Ciabatta, Pita, Bagel, English Muffin$\}$
- item_value and resource_requirement for each bread type are:

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

And the total storage capacity is 180 units.