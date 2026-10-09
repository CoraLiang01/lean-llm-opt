Let $I$ be the set of items in value.csv, with $I = \{1,2,\ldots,140\}$, and for each $i \in I$, let $v_i$ and $w_i$ be the value and weight of item $i$ from columns "value" and "weight" in table_id file_0_view_0.

Decision variables:
For each $i \in I$, $x_i \in \{0,1\}$, where $x_i = 1$ if item $i$ is selected, $0$ otherwise.

Objective:
Maximize $\sum_{i \in I} v_i x_i$

Subject to:
$\sum_{i \in I} w_i x_i \leq 15$

$x_i \in \{0,1\}$ for all $i \in I$

Data Mapping:
Set $I$ and parameters $v_i$, $w_i$ are from table_id file_0_view_0, columns "item", "value", "weight". The weight limit $15$ is from the user description.