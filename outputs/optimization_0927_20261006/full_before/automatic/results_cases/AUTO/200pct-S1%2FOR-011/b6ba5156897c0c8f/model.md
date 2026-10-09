Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the following products:

- Spinach
- Shiitake Mushrooms
- Apples
- Carrots
- Basil
- Potatoes
- Green Beans
- Blueberries
- Oranges
- Watermelons

Let $v_i$ be the Value per unit and $w_i$ the Weight per unit for each product $i$, as given below:

| Product Name         | $v_i$ (Value) | $w_i$ (Weight) |
|----------------------|:-------------:|:--------------:|
| Spinach              |      64       |      230       |
| Shiitake Mushrooms   |      75       |      637       |
| Apples               |      68       |      773       |
| Carrots              |      11       |      653       |
| Basil                |      91       |      755       |
| Potatoes             |      31       |      670       |
| Green Beans          |      90       |      505       |
| Blueberries          |      56       |      821       |
| Oranges              |      10       |       83       |
| Watermelons          |      24       |      249       |

The total stock capacity is $C = 875$.

The mathematical model is:

$$
\begin{align*}
\max \quad & 64x_{\text{Spinach}} + 75x_{\text{Shiitake Mushrooms}} + 68x_{\text{Apples}} + 11x_{\text{Carrots}} + 91x_{\text{Basil}} \\
          & + 31x_{\text{Potatoes}} + 90x_{\text{Green Beans}} + 56x_{\text{Blueberries}} + 10x_{\text{Oranges}} + 24x_{\text{Watermelons}} \\[2ex]
\text{s.t.} \quad
& 230x_{\text{Spinach}} + 637x_{\text{Shiitake Mushrooms}} + 773x_{\text{Apples}} + 653x_{\text{Carrots}} + 755x_{\text{Basil}} \\
& \quad + 670x_{\text{Potatoes}} + 505x_{\text{Green Beans}} + 821x_{\text{Blueberries}} + 83x_{\text{Oranges}} + 249x_{\text{Watermelons}} \leq 875 \\[2ex]
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\end{align*}
$$

Where each $x_i$ is the number of units of product $i$ to order each day, and must be a nonnegative integer.