##### Sets and Indices

Let $I$ be the set of bread types, indexed by $i$.

##### Parameters (from products.csv and capacity.csv, preserving identifiers and coefficients)

For each bread type $i$ (using item_name):

- $p_i$: expected profit per unit (item_value)
- $a_i$: storage space required per unit (resource_requirement)

Total available storage capacity:

- $C = 180$ (resource_capacity from capacity.csv, archive_revision_number=3, archived_attachment_count=3)

##### Decision Variables

For each bread type $i$:

- $x_i$: number of units of bread type $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

##### Mathematical Model

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
\]

**Subject to:**

\[
\sum_{i \in I} a_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

##### Data

Bread types and parameters (in source order):

| item_name        | item_value ($p_i$) | resource_requirement ($a_i$) |
|------------------|-------------------|------------------------------|
| Baguette         | 888               | 4                            |
| Croissant        | 134               | 2                            |
| Sourdough        | 129               | 4                            |
| Rye Bread        | 370               | 3                            |
| Brioche          | 921               | 2                            |
| Focaccia         | 765               | 1                            |
| Ciabatta         | 154               | 2                            |
| Pita             | 837               | 1                            |
| Bagel            | 584               | 3                            |
| English Muffin   | 365               | 3                            |

Storage capacity:

\[
C = 180
\]

##### Complete Model (with explicit coefficients):

\[
\max \big(
888\,x_{\text{Baguette}} + 134\,x_{\text{Croissant}} + 129\,x_{\text{Sourdough}} + 370\,x_{\text{Rye Bread}} + 921\,x_{\text{Brioche}} + 765\,x_{\text{Focaccia}} + 154\,x_{\text{Ciabatta}} + 837\,x_{\text{Pita}} + 584\,x_{\text{Bagel}} + 365\,x_{\text{English Muffin}}
\big)
\]

Subject to:

\[
4\,x_{\text{Baguette}} + 2\,x_{\text{Croissant}} + 4\,x_{\text{Sourdough}} + 3\,x_{\text{Rye Bread}} + 2\,x_{\text{Brioche}} + 1\,x_{\text{Focaccia}} + 2\,x_{\text{Ciabatta}} + 1\,x_{\text{Pita}} + 3\,x_{\text{Bagel}} + 3\,x_{\text{English Muffin}} \leq 180
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{Baguette}, \text{Croissant}, \text{Sourdough}, \text{Rye Bread}, \text{Brioche}, \text{Focaccia}, \text{Ciabatta}, \text{Pita}, \text{Bagel}, \text{English Muffin}\}
\]