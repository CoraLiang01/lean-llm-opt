Let $x_i$ be the number of units of product $i$ (where $i$ indexes the following FAUX products) to be fulfilled.

Maximize total revenue:
$$
\max \Big[
35.9\, x_{\text{FAUX FUR JEWEL SWEATER}}
+ 69.9\, x_{\text{FAUX LEATHER BOMBER JACKET}}
+ 99.9\, x_{\text{FAUX LEATHER BOXY FIT JACKET}}
+ 99.9\, x_{\text{FAUX LEATHER JACKET}}
+ 159.0\, x_{\text{FAUX LEATHER OVERSIZED JACKET LIMITED EDITION}}
+ 69.99\, x_{\text{FAUX LEATHER PUFFER JACKET}}
+ 99.9\, x_{\text{FAUX SHEARLING LINED SUEDE BOOTS}}
+ 89.9\, x_{\text{FAUX SHEARLING PLAID JACKET}}
+ 69.9\, x_{\text{FAUX SUEDE BOMBER JACKET}}
+ 89.9\, x_{\text{FAUX SUEDE JACKET}}
+ 69.9\, x_{\text{FAUX SUEDE OVERSHIRT}}
+ 89.9\, x_{\text{FAUX SUEDE PATCH JACKET}}
\Big]
$$

Subject to, for each product $i$:

- Demand constraint:
  $$
  x_i \leq \text{Demand}_i
  $$
- Inventory constraint:
  $$
  x_i \leq \text{Initial Inventory}_i
  $$
- Nonnegativity and integrality:
  $$
  x_i \in \mathbb{Z}_{\geq 0}
  $$

Where:

| Product Name                                         | Revenue | Demand | Initial Inventory |
|------------------------------------------------------|---------|--------|------------------|
| FAUX FUR JEWEL SWEATER                               | 35.9    | 3025   | 20970            |
| FAUX LEATHER BOMBER JACKET                           | 69.9    | 9585   | 71970            |
| FAUX LEATHER BOXY FIT JACKET                         | 99.9    | 4486   | 32730            |
| FAUX LEATHER JACKET                                  | 99.9    | 10322  | 71130            |
| FAUX LEATHER OVERSIZED JACKET LIMITED EDITION        | 159.0   | 4868   | 34910            |
| FAUX LEATHER PUFFER JACKET                           | 69.99   | 8482   | 64010            |
| FAUX SHEARLING LINED SUEDE BOOTS                     | 99.9    | 2607   | 20760            |
| FAUX SHEARLING PLAID JACKET                          | 89.9    | 1784   | 12490            |
| FAUX SUEDE BOMBER JACKET                             | 69.9    | 6626   | 50300            |
| FAUX SUEDE JACKET                                    | 89.9    | 3256   | 24570            |
| FAUX SUEDE OVERSHIRT                                 | 69.9    | 2955   | 24430            |
| FAUX SUEDE PATCH JACKET                              | 89.9    | 910    | 7070             |

For each $i$, $\text{Demand}_i$ and $\text{Initial Inventory}_i$ are as above.