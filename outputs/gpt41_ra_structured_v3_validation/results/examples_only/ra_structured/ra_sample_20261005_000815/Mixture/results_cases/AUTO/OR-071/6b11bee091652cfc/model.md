Let  
- $x_i$ = production quantity (in units) of product $i$ (continuous, $x_i \geq 0$), for each product $i$ listed in the table below.

**Parameters (from 41.csv):**  
For each product $i$:
- $l_i$ = Labor per unit required for product $i$
- $m_i$ = Material per unit required for product $i$
- $s_i$ = Selling price per unit of product $i$
- $v_i$ = Variable cost per unit of product $i$

**Constants:**
- Total weekly labor capacity: $L = 1650$
- Total weekly material capacity: $M = 1850$
- Fixed weekly operating cost: $F = 4500$

---

### Mathematical Model

**Objective:**  
Maximize weekly net profit (total sales revenue minus total variable cost minus fixed weekly operating cost):

\[
\max_{x_i \geq 0} \left[ \sum_{i} (s_i - v_i) x_i - F \right]
\]

**Subject to:**

1. **Labor capacity constraint:**
   \[
   \sum_{i} l_i x_i \leq L
   \]

2. **Material capacity constraint:**
   \[
   \sum_{i} m_i x_i \leq M
   \]

3. **Nonnegativity:**
   \[
   x_i \geq 0 \quad \forall i
   \]

---

#### Where the products and their coefficients are:

| Product Name                      | $l_i$ (Labor) | $m_i$ (Material) | $s_i$ (Selling Price) | $v_i$ (Variable Cost) |
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
| Linen Blend Shirt                 | 3.2           | 4.3              | 127                   | 64                    |
| Graphic Print Tee                 | 2             | 3                | 82                    | 40                    |
| Quilted Vest                      | 6.8           | 7                | 205                   | 105                   |
| Chambray Work Shirt               | 3             | 4.1              | 122                   | 61                    |
| Microfiber Sport Shirt            | 2.3           | 3.3              | 88                    | 45                    |
| Wool Blend Peacoat                | 7.2           | 7.5              | 215                   | 115                   |
| Seersucker Short-sleeve           | 2.7           | 3.6              | 93                    | 47                    |
| Twill Utility Shirt               | 3.3           | 4.5              | 130                   | 67                    |
| Basic Scoop Neck Tee              | 2.2           | 3.2              | 84                    | 42                    |
| Waffle Knit Pullover              | 6.1           | 6.3              | 187                   | 91                    |
| Button-Down Poplin Shirt          | 3.6           | 4.7              | 135                   | 70                    |
| Slub Cotton T-Shirt               | 2.6           | 3.5              | 92                    | 49                    |
| Drawstring Cargo Pants            | 6.4           | 6.6              | 192                   | 93                    |
| Pima Cotton Dress Shirt           | 3.8           | 4.9              | 140                   | 73                    |
| Pocket T-Shirt                    | 2.4           | 3.3              | 87                    | 44                    |
| Fleece-Lined Track Suit           | 7             | 7.2              | 210                   | 110                   |
| Printed Camp Collar Shirt         | 2.9           | 4                | 120                   | 60                    |
| Performance Dry-Fit Tee           | 2             | 3                | 80                    | 39                    |
| Corduroy Shacket                  | 6.7           | 6.9              | 202                   | 97                    |
| Herringbone Casual Shirt          | 3             | 4.1              | 123                   | 62                    |
| Organic Cotton Tee                | 2.1           | 3.1              | 85                    | 43                    |
| Cashmere Blend Sweater            | 7.5           | 7.8              | 225                   | 125                   |
| Striped Rugby Shirt               | 3.4           | 4.4              | 129                   | 66                    |
| Raglan Sleeve Tee                 | 2.6           | 3.6              | 94                    | 51                    |
| Waterproof Rain Jacket            | 7.1           | 7.4              | 212                   | 112                   |
| Micro-check Business Shirt        | 3.7           | 4.8              | 138                   | 72                    |
| Heather Grey T-Shirt              | 2.3           | 3.2              | 86                    | 45                    |
| Slim Fit Chinos                   | 6.3           | 6.5              | 190                   | 90                    |
| Band Collar Shirt                 | 3.1           | 4.2              | 126                   | 64                    |
| Triblend Jersey Tee               | 2.5           | 3.5              | 91                    | 49                    |
| Sherpa Lined Pullover             | 6.9           | 7.1              | 208                   | 108                   |
| Western Style Shirt               | 3.2           | 4.3              | 128                   | 65                    |
| Recycled Poly T-Shirt             | 2.7           | 3.7              | 96                    | 52                    |
| Insulated Puffer Jacket           | 7.4           | 7.7              | 220                   | 120                   |
| End-on-End Fabric Shirt           | 3.5           | 4.6              | 134                   | 69                    |
| Oversized Graphic Tee             | 2.8           | 3.8              | 98                    | 54                    |
| Stretch Hybrid Shorts             | 6             | 6.2              | 186                   | 89                    |
| Non-Iron Travel Shirt             | 3.9           | 5                | 142                   | 75                    |
| Sun-Protective T-Shirt            | 2.1           | 3                | 81                    | 40                    |
| Velvet Bomber Jacket              | 7.3           | 7.6              | 218                   | 118                   |
| Madras Plaid Shirt                | 3.3           | 4.4              | 129                   | 66                    |
| Tie-Dye T-Shirt                   | 2.4           | 3.3              | 89                    | 46                    |
| Lightweight Windbreaker           | 5.8           | 6                | 180                   | 85                    |
| Pinpoint Oxford Shirt             | 3.6           | 4.7              | 136                   | 71                    |
| Bamboo Viscose Tee                | 2             | 2.9              | 79                    | 38                    |
| Tapered Denim Jeans               | 6.6           | 6.9              | 198                   | 96                    |
| Contrast Collar Shirt             | 3.8           | 4.9              | 141                   | 74                    |
| Mock Neck Long-sleeve             | 2.2           | 3.1              | 86                    | 43                    |
| Down-filled Parka                 | 7.8           | 8.1              | 235                   | 135                   |
| Textured Grid Shirt               | 3.1           | 4.2              | 125                   | 63                    |
| Heavy Cotton T-Shirt              | 2.5           | 3.4              | 90                    | 48                    |
| Convertible Zip-Off Trousers      | 6.5           | 6.8              | 195                   | 94                    |
| Tencel Blend Casual Shirt         | 3.4           | 4.5              | 131                   | 67                    |
| Relaxed Fit Tee                   | 2.7           | 3.6              | 95                    | 51                    |
| Faux Leather Jacket               | 7             | 7.3              | 210                   | 111                   |
| Speckled Slub Shirt               | 3             | 4.1              | 124                   | 62                    |
| Knit Cardigan Sweater             | 6.1           | 6.3              | 188                   | 90                    |
| Two-Pocket Utility Shirt          | 3.7           | 4.8              | 139                   | 72                    |
| Performance Stretch Tee           | 2.6           | 3.5              | 93                    | 50                    |
| Technical Shell Jacket            | 7.6           | 7.9              | 230                   | 130                   |
| Semi-Spread Collar Shirt          | 3.2           | 4.3              | 127                   | 64                    |
| Washed Pigment Dye Tee            | 2.1           | 3                | 82                    | 40                    |
| Twill Blazer (Unlined)            | 6.9           | 7.1              | 205                   | 100                   |
| Brushed Twill Shirt               | 3.5           | 4.6              | 134                   | 69                    |
| Supima Cotton Tee                 | 2.8           | 3.7              | 97                    | 53                    |
| High-Waist Trousers               | 6.2           | 6.4              | 189                   | 91                    |
| Printed Geometric Shirt           | 3.9           | 5                | 143                   | 76                    |
| Minimalist Plain Tee              | 2             | 2.9              | 78                    | 37                    |
| Reversible Quilted Jacket         | 7.4           | 7.7              | 222                   | 122                   |
| Dobby Weave Shirt                 | 3.3           | 4.4              | 130                   | 67                    |
| Heavyweight Boxy Tee              | 2.4           | 3.3              | 89                    | 46                    |
| Casual Wool Overcoat              | 8             | 8.3              | 240                   | 140                   |
| Half-Zip Pullover Shirt           | 3.6           | 4.7              | 137                   | 71                    |
| Drop Shoulder T-Shirt             | 2.2           | 3.1              | 87                    | 44                    |
| Straight Leg Corduroys            | 6.7           | 7                | 200                   | 99                    |
| Camp Collar Floral Shirt          | 2.9           | 4                | 121                   | 61                    |
| Garment Dyed Tee                  | 2.5           | 3.4              | 90                    | 47                    |
| Suede Bomber Jacket               | 7.7           | 8                | 232                   | 132                   |
| Melange Knit Shirt                | 3.1           | 4.2              | 126                   | 64                    |
| Longline T-Shirt                  | 2.6           | 3.5              | 94                    | 51                    |
| Wide Leg Culottes                 | 6.3           | 6.5              | 191                   | 92                    |
| Tropical Print Shirt              | 3.8           | 4.9              | 140                   | 73                    |
| Distressed Cotton Tee             | 2.3           | 3.2              | 88                    | 45                    |
| Patchwork Denim Jacket            | 7.2           | 7.5              | 215                   | 114                   |
| Micro-Dot Shirt                   | 3             | 4.1              | 123                   | 62                    |
| Muscle Fit T-Shirt                | 2.1           | 3                | 83                    | 41                    |
| Woven Utility Vest                | 6.4           | 6.6              | 193                   | 93                    |
| Silk Blend Dress Shirt            | 4             | 5.1              | 145                   | 78                    |
| Heavyweight Jersey T-Shirt        | 2.7           | 3.6              | 96                    | 52                    |
| Slim Fit Ankle Grazer             | 6             | 6.2              | 186                   | 88                    |
| Lattice Weave Shirt               | 3.2           | 4.3              | 128                   | 65                    |
| Boxy Cropped Tee                  | 2.2           | 3.1              | 84                    | 42                    |
| Reversible Bomber Jacket          | 7.5           | 7.8              | 225                   | 124                   |
| Stretch Performance Shirt         | 3.5           | 4.6              | 133                   | 68                    |
| Sublimation Print Tee             | 2.4           | 3.3              | 89                    | 46                    |
| Brushed Cotton Trousers           | 6.6           | 6.9              | 198                   | 97                    |
| Micro-Waffle Knit Shirt           | 3.1           | 4.2              | 125                   | 63                    |
| Reflective Detail Tee             | 2             | 2.9              | 80                    | 39                    |
| Shearling Lined Coat              | 7.9           | 8.2              | 238                   | 138                   |
| Printed Patchwork Shirt           | 3.4           | 4.5              | 130                   | 66                    |
| Washed Effect T-Shirt             | 2.6           | 3.5              | 92                    | 49                    |
| Velour Track Pants                | 5.8           | 6                | 182                   | 86                    |
| Fine Wale Corduroy Shirt          | 3.7           | 4.8              | 138                   | 72                    |
| Organic Hemp T-Shirt              | 2.8           | 3.7              | 96                    | 52                    |
| Leather Biker Jacket              | 8.2           | 8.5              | 250                   | 150                   |
| Supersoft Modal Shirt             | 3             | 4.1              | 123                   | 61                    |
| Deep V-Neck Tee                   | 2.3           | 3.2              | 87                    | 44                    |
| Wool Blend Trousers               | 6.8           | 7.1              | 203                   | 103                   |
| Contrast Stitch Shirt             | 3.9           | 5                | 142                   | 75                    |
| Embroidered Pocket Tee            | 2.5           | 3.4              | 90                    | 47                    |
| Camo Print Hoodie                 | 6.1           | 6.3              | 187                   | 89                    |
| Silk-Touch Poplin Shirt           | 3.3           | 4.4              | 129                   | 66                    |
| Slim Fit Pique Polo               | 2.7           | 3.6              | 94                    | 50                    |
| Technical Cargo Vest              | 6.9           | 7.2              | 206                   | 106                   |
| Cutaway Collar Shirt              | 3.6           | 4.7              | 136                   | 70                    |
| Lycra Blend Active Tee            | 2.1           | 3                | 81                    | 40                    |
| Heavy Knit Sweater                | 7.3           | 7.6              | 218                   | 117                   |
| Textured Stripe Shirt             | 3.2           | 4.3              | 127                   | 64                    |
| Bio-Washed T-Shirt                | 2.8           | 3.7              | 97                    | 53                    |
| Mohair Blend Cardigan             | 7             | 7.3              | 211                   | 110                   |
| Shantung Silk Shirt               | 4.1           | 5.2              | 148                   | 80                    |
| Distressed Denim Tee              | 2             | 2.9              | 79                    | 38                    |
| High-Tech Ski Jacket              | 8.5           | 8.8              | 260                   | 160                   |
| Fine Poplin Check Shirt           | 3.5           | 4.6              | 133                   | 68                    |
| Slim Fit Linen T-Shirt            | 2.2           | 3.1              | 85                    | 43                    |
| Ripstop Utility Pants             | 6.3           | 6.5              | 190                   | 91                    |
| Classic Barrel Cuff Shirt         | 3.7           | 4.8              | 139                   | 73                    |
| Bamboo Fiber Tee                  | 2.4           | 3.3              | 89                    | 46                    |
| Shearling Trench Coat             | 8.1           | 8.4              | 245                   | 145                   |
| Slim Fit Chambray                 | 3             | 4.1              | 124                   | 62                    |
| Printed Geometric Tee             | 2.5           | 3.4              | 91                    | 48                    |
| Wool Cargo Trousers               | 6.7           | 7                | 201                   | 101                   |
| Two-Tone Oxford Shirt             | 3.3           | 4.4              | 130                   | 67                    |
| Slogan Print T-Shirt              | 2.6           | 3.5              | 93                    | 50                    |
| Trench Coat (Lined)               | 8.3           | 8.6              | 255                   | 155                   |
| Spread Collar Shirt               | 3.8           | 4.9              | 140                   | 73                    |
| Printed Stripe Tee                | 2.7           | 3.6              | 97                    | 53                    |
| Non-Iron Twill Shirt              | 3.1           | 4.2              | 126                   | 64                    |
| Recycled Material Tee             | 2             | 2.9              | 79                    | 38                    |
| Faux Fur Coat                     | 7.7           | 8                | 235                   | 134                   |
| Seersucker Popover Shirt          | 3.4           | 4.5              | 131                   | 67                    |
| Graphic Back Print Tee            | 2.4           | 3.3              | 89                    | 46                    |
| Corduroy Field Jacket             | 6.7           | 7                | 200                   | 98                    |
| Linen/Cotton Blend Shirt          | 3.6           | 4.7              | 136                   | 71                    |
| Lightweight Jersey Tee            | 2.5           | 3.4              | 91                    | 48                    |
| High Performance Hiking Pants     | 6.5           | 6.8              | 194                   | 94                    |
| Cuban Collar Shirt                | 3.9           | 5                | 142                   | 76                    |
| Faded Pigment Tee                 | 2.8           | 3.7              | 98                    | 54                    |
| Canvas Chore Coat                 | 7.2           | 7.5              | 216                   | 116                   |
| Brushed Cotton Shirt              | 3.2           | 4.3              | 127                   | 65                    |
| Ribbed Knit T-Shirt               | 2.1           | 3                | 83                    | 41                    |
| Wide Leg Trousers                 | 6.1           | 6.3              | 187                   | 89                    |
| Patchwork Denim Shirt             | 3.5           | 4.6              | 133                   | 69                    |
| Basic White T-Shirt               | 2.3           | 3.2              | 87                    | 44                    |
| Mohair Knit Sweater               | 7.5           | 7.8              | 224                   | 123                   |
| Contrast Stitch Polo              | 2.9           | 3.8              | 99                    | 55                    |
| Tailored Wool Trousers            | 6.6           | 6.9              | 200                   | 99                    |
| Heavyweight Denim Shirt           | 4             | 5.1              | 145                   | 77                    |

---

**Summary:**  
- Decision variables: $x_i \geq 0$ (continuous), for each product $i$.
- Objective: Maximize $\sum_i (s_i - v_i)x_i - 4500$
- Subject to: $\sum_i l_i x_i \leq 1650$, $\sum_i m_i x_i \leq 1850$, $x_i \geq 0$ for all $i$.
- All coefficients and product names are as listed above, in the original CSV order.