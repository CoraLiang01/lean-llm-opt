Let $x_i$ be the integer number of units of product $i$ to order daily, for each $i$ in the set of products below.

Maximize total benefit:
$$
\max\ 49x_{\text{Spinach}} + 30x_{\text{Shiitake Mushrooms}} + 30x_{\text{Apples}} + 18x_{\text{Carrots}} + 54x_{\text{Basil}} + 27x_{\text{Potatoes}} + 91x_{\text{Green Beans}} + 88x_{\text{Blueberries}} + 78x_{\text{Oranges}} + 22x_{\text{Watermelons}}
$$

Subject to the inventory capacity constraint:
$$
282x_{\text{Spinach}} + 83x_{\text{Shiitake Mushrooms}} + 251x_{\text{Apples}} + 257x_{\text{Carrots}} + 88x_{\text{Basil}} + 52x_{\text{Potatoes}} + 198x_{\text{Green Beans}} + 203x_{\text{Blueberries}} + 87x_{\text{Oranges}} + 265x_{\text{Watermelons}} \leq 1035
$$

and integrality and nonnegativity:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for all products } i
$$

Where the products, weights, and values are:

| Product Name         | Weight | Value |
|---------------------|--------|-------|
| Spinach             | 282    | 49    |
| Shiitake Mushrooms  | 83     | 30    |
| Apples              | 251    | 30    |
| Carrots             | 257    | 18    |
| Basil               | 88     | 54    |
| Potatoes            | 52     | 27    |
| Green Beans         | 198    | 91    |
| Blueberries         | 203    | 88    |
| Oranges             | 87     | 78    |
| Watermelons         | 265    | 22    |