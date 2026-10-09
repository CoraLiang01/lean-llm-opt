#### Sets and Indices

Let $I$ be the set of bread types, indexed by $i$.

#### Parameters

From products.csv (in source order):

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

From capacity.csv:

- resource_capacity = 180

#### Decision Variables

For each bread type $i \in I$:

- $x_i$: number of units of bread type $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

#### Mathematical Model

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
\]
where $p_i$ is the item_value for bread $i$.

**Constraint:**
\[
\sum_{i \in I} a_i x_i \leq 180
\]
where $a_i$ is the resource_requirement for bread $i$.

**Variable Domains:**
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\]

#### Explicit Formulation

Let the bread types be indexed in the order given above ($i=1$ for Baguette, $i=2$ for Croissant, etc.).

**Objective:**
\[
\max \Big(
888\,x_{\text{Baguette}} + 134\,x_{\text{Croissant}} + 129\,x_{\text{Sourdough}} + 370\,x_{\text{Rye Bread}} + 921\,x_{\text{Brioche}} + 765\,x_{\text{Focaccia}} + 154\,x_{\text{Ciabatta}} + 837\,x_{\text{Pita}} + 584\,x_{\text{Bagel}} + 365\,x_{\text{English Muffin}}
\Big)
\]

**Constraint:**
\[
4\,x_{\text{Baguette}} + 2\,x_{\text{Croissant}} + 4\,x_{\text{Sourdough}} + 3\,x_{\text{Rye Bread}} + 2\,x_{\text{Brioche}} + 1\,x_{\text{Focaccia}} + 2\,x_{\text{Ciabatta}} + 1\,x_{\text{Pita}} + 3\,x_{\text{Bagel}} + 3\,x_{\text{English Muffin}} \leq 180
\]

**Variable Domains:**
\[
x_{\text{Baguette}},\ x_{\text{Croissant}},\ x_{\text{Sourdough}},\ x_{\text{Rye Bread}},\ x_{\text{Brioche}},\ x_{\text{Focaccia}},\ x_{\text{Ciabatta}},\ x_{\text{Pita}},\ x_{\text{Bagel}},\ x_{\text{English Muffin}} \in \mathbb{Z}_{\geq 0}
\]