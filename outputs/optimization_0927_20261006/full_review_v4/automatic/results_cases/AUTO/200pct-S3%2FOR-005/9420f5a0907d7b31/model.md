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

The parameters for each bread type $i$ are:

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

The total available storage capacity is:

- resource_capacity = 180

##### Mathematical Model

**Decision variables:**
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for each bread type } i
$$

**Objective:**
$$
\max \left(
888\,x_{\text{Baguette}} +
134\,x_{\text{Croissant}} +
129\,x_{\text{Sourdough}} +
370\,x_{\text{Rye Bread}} +
921\,x_{\text{Brioche}} +
765\,x_{\text{Focaccia}} +
154\,x_{\text{Ciabatta}} +
837\,x_{\text{Pita}} +
584\,x_{\text{Bagel}} +
365\,x_{\text{English Muffin}}
\right)
$$

**Subject to:**
$$
4\,x_{\text{Baguette}}
+ 2\,x_{\text{Croissant}}
+ 4\,x_{\text{Sourdough}}
+ 3\,x_{\text{Rye Bread}}
+ 2\,x_{\text{Brioche}}
+ 1\,x_{\text{Focaccia}}
+ 2\,x_{\text{Ciabatta}}
+ 1\,x_{\text{Pita}}
+ 3\,x_{\text{Bagel}}
+ 3\,x_{\text{English Muffin}}
\leq 180
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

Where each $x_i$ is the number of units of bread type $i$ to order each day.