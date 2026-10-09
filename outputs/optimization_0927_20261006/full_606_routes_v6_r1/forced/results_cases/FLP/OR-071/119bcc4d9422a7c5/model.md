##### Sets and Indices

Let $P$ be the set of all products, indexed by $p$.

##### Parameters

For each product $p \in P$ (from 41.csv):

- $a_p$: labor required per unit of product $p$
- $b_p$: material required per unit of product $p$
- $s_p$: selling price per unit of product $p$
- $v_p$: variable cost per unit of product $p$

Factory-wide parameters:

- Total weekly labor capacity: $L = 1650$
- Total weekly material capacity: $M = 1850$
- Fixed weekly operating cost: $F = 4500$

##### Decision Variables

$x_p \geq 0$: quantity of product $p$ to produce (continuous, for all $p \in P$)

##### Objective Function

\[
\max \left( \sum_{p \in P} (s_p - v_p) x_p - F \right)
\]

##### Constraints

1. Labor capacity:
   \[
   \sum_{p \in P} a_p x_p \leq L
   \]
2. Material capacity:
   \[
   \sum_{p \in P} b_p x_p \leq M
   \]
3. Nonnegativity:
   \[
   x_p \geq 0 \quad \forall p \in P
   \]

##### Parameters from 41.csv

$P$ = {Classic Oxford Shirt, Cotton Crew Neck T-Shirt, French Terry Hoodie, Denim Snap Shirt, Jersey V-Neck T-Shirt, Unisex Jogger Pants, Flannel Plaid Shirt, Performance Polo Shirt, Long-sleeve Henley, Heavyweight Sweatshirt, Linen Blend Shirt, Graphic Print Tee, Quilted Vest, Chambray Work Shirt, Microfiber Sport Shirt, Wool Blend Peacoat, Seersucker Short-sleeve, Twill Utility Shirt, Basic Scoop Neck Tee, Waffle Knit Pullover, Button-Down Poplin Shirt, Slub Cotton T-Shirt, Drawstring Cargo Pants, Pima Cotton Dress Shirt, Pocket T-Shirt, Fleece-Lined Track Suit, Printed Camp Collar Shirt, Performance Dry-Fit Tee, Corduroy Shacket, Herringbone Casual Shirt, Organic Cotton Tee, Cashmere Blend Sweater, Striped Rugby Shirt, Raglan Sleeve Tee, Waterproof Rain Jacket, Micro-check Business Shirt, Heather Grey T-Shirt, Slim Fit Chinos, Band Collar Shirt, Triblend Jersey Tee, Sherpa Lined Pullover, Western Style Shirt, Recycled Poly T-Shirt, Insulated Puffer Jacket, End-on-End Fabric Shirt, Oversized Graphic Tee, Stretch Hybrid Shorts, Non-Iron Travel Shirt, Sun-Protective T-Shirt, Velvet Bomber Jacket, Madras Plaid Shirt, Tie-Dye T-Shirt, Lightweight Windbreaker, Pinpoint Oxford Shirt, Bamboo Viscose Tee, Tapered Denim Jeans, Contrast Collar Shirt, Mock Neck Long-sleeve, Down-filled Parka, Textured Grid Shirt, Heavy Cotton T-Shirt, Convertible Zip-Off Trousers, Tencel Blend Casual Shirt, Relaxed Fit Tee, Faux Leather Jacket, Speckled Slub Shirt, Embroidered T-Shirt, Knit Cardigan Sweater, Two-Pocket Utility Shirt, Performance Stretch Tee, Technical Shell Jacket, Semi-Spread Collar Shirt, Washed Pigment Dye Tee, Twill Blazer (Unlined), Brushed Twill Shirt, Supima Cotton Tee, High-Waist Trousers, Printed Geometric Shirt, Minimalist Plain Tee, Reversible Quilted Jacket, Dobby Weave Shirt, Heavyweight Boxy Tee, Casual Wool Overcoat, Half-Zip Pullover Shirt, Drop Shoulder T-Shirt, Straight Leg Corduroys, Camp Collar Floral Shirt, Garment Dyed Tee, Suede Bomber Jacket, Melange Knit Shirt, Longline T-Shirt, Wide Leg Culottes, Tropical Print Shirt, Distressed Cotton Tee, Patchwork Denim Jacket, Micro-Dot Shirt, Muscle Fit T-Shirt, Woven Utility Vest, Silk Blend Dress Shirt, Heavyweight Jersey T-Shirt, Slim Fit Ankle Grazer, Lattice Weave Shirt, Boxy Cropped Tee, Reversible Bomber Jacket, Stretch Performance Shirt, Sublimation Print Tee, Brushed Cotton Trousers, Micro-Waffle Knit Shirt, Reflective Detail Tee, Shearling Lined Coat, Printed Patchwork Shirt, Washed Effect T-Shirt, Velour Track Pants, Fine Wale Corduroy Shirt, Organic Hemp T-Shirt, Leather Biker Jacket, Supersoft Modal Shirt, Deep V-Neck Tee, Wool Blend Trousers, Contrast Stitch Shirt, Embroidered Pocket Tee, Camo Print Hoodie, Silk-Touch Poplin Shirt, Slim Fit Pique Polo, Technical Cargo Vest, Cutaway Collar Shirt, Lycra Blend Active Tee, Heavy Knit Sweater, Textured Stripe Shirt, Bio-Washed T-Shirt, Mohair Blend Cardigan, Shantung Silk Shirt, Distressed Denim Tee, High-Tech Ski Jacket, Fine Poplin Check Shirt, Slim Fit Linen T-Shirt, Ripstop Utility Pants, Classic Barrel Cuff Shirt, Bamboo Fiber Tee, Shearling Trench Coat, Slim Fit Chambray, Printed Geometric Tee, Wool Cargo Trousers, Two-Tone Oxford Shirt, Slogan Print T-Shirt, Water-Resistant Shell, Grandad Collar Shirt, Embroidered Logo Tee, Casual Knit Blazer, Eco-Friendly Hemp Shirt, Cotton Slub Polo, Corduroy Carpenter Pants, Linen Safari Shirt, Ribbed Crew Neck Tee, Hybrid Hiking Trousers, Brushed Flannel Shirt, Microfiber V-Neck, Down-Alternative Vest, Checked Seersucker Shirt, Fitted Basic Tee, Heavyweight Denim Jacket, Woven Stripe Shirt, Tech Fleece Pullover, Pin Dot Print Shirt, Supersoft Modal T-Shirt, Water-Resistant Chinos, Tailored Fit Dress Shirt, Printed Art Graphic Tee, Cashmere Cable Knit, Plaid Flannel Overshirt, Moisture-Wicking Polo, Slim Tapered Jeans, Contrast Cuff Shirt, Sustainable Cotton Tee, Trench Coat (Lined), Spread Collar Shirt, Printed Stripe Tee, Non-Iron Twill Shirt, Recycled Material Tee, Faux Fur Coat, Seersucker Popover Shirt, Graphic Back Print Tee, Corduroy Field Jacket, Linen/Cotton Blend Shirt, Lightweight Jersey Tee, High Performance Hiking Pants, Cuban Collar Shirt, Faded Pigment Tee, Canvas Chore Coat, Brushed Cotton Shirt, Ribbed Knit T-Shirt, Wide Leg Trousers, Patchwork Denim Shirt, Basic White T-Shirt, Mohair Knit Sweater, Contrast Stitch Polo, Tailored Wool Trousers, Heavyweight Denim Shirt}

For each $p \in P$:

| Product Name                      | $a_p$ (Labor) | $b_p$ (Material) | $s_p$ (Selling Price) | $v_p$ (Variable Cost) |
|-----------------------------------|:-------------:|:----------------:|:---------------------:|:---------------------:|
| Classic Oxford Shirt              | 3.1           | 4.2              | 125                   | 63                    |
| Cotton Crew Neck T-Shirt          | 2.1           | 3.1              | 83                    | 41                    |
| French Terry Hoodie               | 6.5           | 6.8              | 195                   | 95                    |
| Denim Snap Shirt                  | 3.4           | 4.6              | 132                   | 68                    |
| Jersey V-Neck T-Shirt             | 2.5           | 3.4              | 90                    | 48                    |
| Unisex Jogger Pants               | 5.9           | 6.1              | 185                   | 88                    |
| Flannel Plaid Shirt               | 3.5           | 4                | 128                   | 65                    |
| Performance Polo Shirt            | 2.9           | 3.9              | 118                   | 57                    |
| Long-sleeve Henley                | 2.8           | 3.7              | 95                    | 50                    |
| Heavyweight Sweatshirt            | 6.2           | 6.4              | 190                   | 92                    |
| ...                               | ...           | ...              | ...                   | ...                   |

(And so on for all products as listed in the retrieved data.)

##### Complete Model

\[
\begin{align*}
\max_{x_p \geq 0} \quad & \sum_{p \in P} (s_p - v_p) x_p - 4500 \\
\text{s.t.} \quad
& \sum_{p \in P} a_p x_p \leq 1650 \\
& \sum_{p \in P} b_p x_p \leq 1850 \\
& x_p \geq 0 \quad \forall p \in P
\end{align*}
\]

where all parameters $a_p$, $b_p$, $s_p$, $v_p$ are as given above for each product $p \in P$.