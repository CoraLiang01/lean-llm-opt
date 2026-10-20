##### Objective Function  

$\quad \quad \max \Bigl(\sum_{i=1}^{198} (SellingPrice_i-VariableCost_i)\,x_i \;-\; 4500\Bigr)$  

##### Constraints  

###### 1. Labor Capacity  

$\quad \quad \sum_{i=1}^{198} Labor_i\,x_i \;\le\; 1650$  

###### 2. Material Capacity  

$\quad \quad \sum_{i=1}^{198} Material_i\,x_i \;\le\; 1850$  

###### 3. Non‑negativity and Continuity  

$\quad \quad x_i \;\ge\; 0,\;\; x_i \in \mathbb{R},\;\; \forall i=1,\dots,198$  

##### Data (Product i, Labor, Material, Selling Price, Variable Cost, Profit per unit)  

| i | Product Name | Labor per unit | Material per unit | Selling Price | Variable Cost | Profit per unit |
|---|--------------|----------------|-------------------|---------------|---------------|-----------------|
| 1 | Classic Oxford Shirt | 3.1 | 4.2 | 125 | 63 | 62 |
| 2 | Cotton Crew Neck T-Shirt | 2.1 | 3.1 | 83 | 41 | 42 |
| 3 | French Terry Hoodie | 6.5 | 6.8 | 195 | 95 | 100 |
| 4 | Denim Snap Shirt | 3.4 | 4.6 | 132 | 68 | 64 |
| 5 | Jersey V-Neck T-Shirt | 2.5 | 3.4 | 90 | 48 | 42 |
| 6 | Unisex Jogger Pants | 5.9 | 6.1 | 185 | 88 | 97 |
| 7 | Flannel Plaid Shirt | 3.5 | 4.0 | 128 | 65 | 63 |
| 8 | Performance Polo Shirt | 2.9 | 3.9 | 118 | 57 | 61 |
| 9 | Long-sleeve Henley | 2.8 | 3.7 | 95 | 50 | 45 |
| 10 | Heavyweight Sweatshirt | 6.2 | 6.4 | 190 | 92 | 98 |
| 11 | Linen Blend Shirt | 3.2 | 4.3 | 127 | 64 | 63 |
| 12 | Graphic Print Tee | 2.0 | 3.0 | 82 | 40 | 42 |
| 13 | Quilted Vest | 6.8 | 7.0 | 205 | 105 | 100 |
| 14 | Chambray Work Shirt | 3.0 | 4.1 | 122 | 61 | 61 |
| 15 | Microfiber Sport Shirt | 2.3 | 3.3 | 88 | 45 | 43 |
| 16 | Wool Blend Peacoat | 7.2 | 7.5 | 215 | 115 | 100 |
| 17 | Seersucker Short-sleeve | 2.7 | 3.6 | 93 | 47 | 46 |
| 18 | Twill Utility Shirt | 3.3 | 4.5 | 130 | 67 | 63 |
| 19 | Basic Scoop Neck Tee | 2.2 | 3.2 | 84 | 42 | 42 |
| 20 | Waffle Knit Pullover | 6.1 | 6.3 | 187 | 91 | 96 |
| 21 | Button-Down Poplin Shirt | 3.6 | 4.7 | 135 | 70 | 65 |
| 22 | Drawstring Cargo Pants | 6.4 | 6.6 | 192 | 93 | 99 |
| 23 | Pima Cotton Dress Shirt | 3.8 | 4.9 | 140 | 73 | 67 |
| 24 | Pocket T-Shirt | 2.4 | 3.3 | 87 | 44 | 43 |
| 25 | Fleece-Lined Track Suit | 7.0 | 7.2 | 210 | 110 | 100 |
| 26 | Printed Camp Collar Shirt | 2.9 | 4.0 | 120 | 60 | 60 |
| 27 | Performance Dry‑Fit Tee | 2.0 | 3.0 | 80 | 39 | 41 |
| 28 | Corduroy Shacket | 6.7 | 6.9 | 202 | 97 | 105 |
| 29 | Herringbone Casual Shirt | 3.0 | 4.1 | 123 | 62 | 61 |
| 30 | Organic Cotton Tee | 2.1 | 3.1 | 85 | 43 | 42 |
| 31 | Cashmere Blend Sweater | 7.5 | 7.8 | 225 | 125 | 100 |
| 32 | Striped Rugby Shirt | 3.4 | 4.4 | 129 | 66 | 63 |
| 33 | Raglan Sleeve Tee | 2.6 | 3.6 | 94 | 51 | 43 |
| 34 | Waterproof Rain Jacket | 7.1 | 7.4 | 212 | 112 | 100 |
| 35 | Micro‑check Business Shirt | 3.7 | 4.8 | 138 | 72 | 66 |
| 36 | Heather Grey T‑Shirt | 2.3 | 3.2 | 86 | 45 | 41 |
| 37 | Slim Fit Chinos | 6.3 | 6.5 | 190 | 90 | 100 |
| 38 | Band Collar Shirt | 3.1 | 4.2 | 126 | 64 | 62 |
| 39 | Triblend Jersey Tee | 2.5 | 3.5 | 91 | 49 | 42 |
| 40 | Sherpa Lined Pullover | 6.9 | 7.1 | 208 | 108 | 100 |
| 41 | Western Style Shirt | 3.2 | 4.3 | 128 | 65 | 63 |
| 42 | Recycled Poly T‑Shirt | 2.7 | 3.7 | 96 | 52 | 44 |
| 43 | Insulated Puffer Jacket | 7.4 | 7.7 | 220 | 120 | 100 |
| 44 | End‑on‑End Fabric Shirt | 3.5 | 4.6 | 134 | 69 | 65 |
| 45 | Oversized Graphic Tee | 2.8 | 3.8 | 98 | 54 | 44 |
| 46 | Stretch Hybrid Shorts | 6.0 | 6.2 | 186 | 89 | 97 |
| 47 | Non‑Iron Travel Shirt | 3.9 | 5.0 | 142 | 75 | 67 |
| 48 | Sun‑Protective T‑Shirt | 2.1 | 3.0 | 81 | 40 | 41 |
| 49 | Velvet Bomber Jacket | 7.3 | 7.6 | 218 | 118 | 100 |
| 50 | Madras Plaid Shirt | 3.3 | 4.4 | 129 | 66 | 63 |
| 51 | Tie‑Dye T‑Shirt | 2.4 | 3.3 | 89 | 46 | 43 |
| 52 | Lightweight Windbreaker | 5.8 | 6.0 | 180 | 85 | 95 |
| 53 | Pinpoint Oxford Shirt | 3.6 | 4.7 | 136 | 71 | 65 |
| 54 | Bamboo Viscose Tee | 2.0 | 2.9 | 79 | 38 | 41 |
| 55 | Tapered Denim Jeans | 6.6 | 6.9 | 198 | 96 | 102 |
| 56 | Contrast Collar Shirt | 3.8 | 4.9 | 141 | 74 | 67 |
| 57 | Mock Neck Long‑sleeve | 2.2 | 3.1 | 86 | 43 | 43 |
| 58 | Down‑filled Parka | 7.8 | 8.1 | 235 | 135 | 100 |
| 59 | Textured Grid Shirt | 3.1 | 4.2 | 125 | 63 | 62 |
| 60 | Heavy Cotton T‑Shirt | 2.5 | 3.4 | 90 | 48 | 42 |
| 61 | Convertible Zip‑Off Trousers | 6.5 | 6.8 | 195 | 94 | 101 |
| 62 | Tencel Blend Casual Shirt | 3.4 | 4.5 | 131 | 67 | 64 |
| 63 | Relaxed Fit Tee | 2.7 | 3.6 | 95 | 51 | 44 |
| 64 | Faux Leather Jacket | 7.0 | 7.3 | 210 | 111 | 99 |
| 65 | Speckled Slub Shirt | 3.0 | 4.1 | 124 | 62 | 62 |
| 66 | Embroidered T‑Shirt | 2.3 | 3.2 | 88 | 45 | 43 |
| 67 | Knit Cardigan Sweater | 6.1 | 6.3 | 188 | 90 | 98 |
| 68 | Two‑Pocket Utility Shirt | 3.7 | 4.8 | 139 | 72 | 67 |
| 69 | Performance Stretch Tee | 2.6 | 3.5 | 93 | 50 | 43 |
| 70 | Technical Shell Jacket | 7.6 | 7.9 | 230 | 130 | 100 |
| 71 | Semi‑Spread Collar Shirt | 3.2 | 4.3 | 127 | 64 | 63 |
| 72 | Washed Pigment Dye Tee | 2.1 | 3.0 | 82 | 40 | 42 |
| 73 | Twill Blazer (Unlined) | 6.9 | 7.1 | 205 | 100 | 105 |
| 74 | Brushed Twill Shirt | 3.5 | 4.6 | 134 | 69 | 65 |
| 75 | Supima Cotton Tee | 2.8 | 3.7 | 97 | 53 | 44 |
| 76 | High‑Waist Trousers | 6.2 | 6.4 | 189 | 91 | 98 |
| 77 | Printed Geometric Shirt | 3.9 | 5.0 | 143 | 76 | 67 |
| 78 | Minimalist Plain Tee | 2.0 | 2.9 | 78 | 37 | 41 |
| 79 | Reversible Quilted Jacket | 7.4 | 7.7 | 222 | 122 | 100 |
| 80 | Dobby Weave Shirt | 3.3 | 4.4 | 130 | 67 | 63 |
| 81 | Heavyweight Boxy Tee | 2.4 | 3.3 | 89 | 46 | 43 |
| 82 | Casual Wool Overcoat | 8.0 | 8.3 | 240 | 140 | 100 |
| 83 | Half‑Zip Pullover Shirt | 3.6 | 4.7 | 137 | 71 | 66 |
| 84 | Drop Shoulder T‑Shirt | 2.2 | 3.1 | 87 | 44 | 43 |
| 85 | Straight Leg Corduroys | 6.7 | 7.0 | 200 | 99 | 101 |
| 86 | Camp Collar Floral Shirt | 2.9 | 4.0 | 121 | 61 | 60 |
| 87 | Garment Dyed Tee | 2.5 | 3.4 | 90 | 47 | 43 |
| 88 | Suede Bomber Jacket | 7.7 | 8.0 | 232 | 132 | 100 |
| 89 | Melange Knit Shirt | 3.1 | 4.2 | 126 | 64 | 62 |
| 90 | Longline T‑Shirt | 2.6 | 3.5 | 94 | 51 | 43 |
| 91 | Wide Leg Culottes | 6.3 | 6.5 | 191 | 92 | 99 |
| 92 | Tropical Print Shirt | 3.8 | 4.9 | 140 | 73 | 67 |
| 93 | Distressed Cotton Tee | 2.3 | 3.2 | 88 | 45 | 43 |
| 94 | Patchwork Denim Jacket | 7.2 | 7.5 | 215 | 114 | 101 |
| 95 | Micro‑Dot Shirt | 3.0 | 4.1 | 123 | 62 | 61 |
| 96 | Muscle Fit T‑Shirt | 2.1 | 3.0 | 83 | 41 | 42 |
| 97 | Woven Utility Vest | 6.4 | 6.6 | 193 | 93 | 100 |
| 98 | Silk Blend Dress Shirt | 4.0 | 5.1 | 145 | 78 | 67 |
| 99 | Heavyweight Jersey T‑Shirt | 2.7 | 3.6 | 96 | 52 | 44 |
| 100 | Slim Fit Ankle Grazer | 6.0 | 6.2 | 186 | 88 | 98 |
| 101 | Lattice Weave Shirt | 3.2 | 4.3 | 128 | 65 | 63 |
| 102 | Boxy Cropped Tee | 2.2 | 3.1 | 84 | 42 | 42 |
| 103 | Reversible Bomber Jacket | 7.5 | 7.8 | 225 | 124 | 101 |
| 104 | Stretch Performance Shirt | 3.5 | 4.6 | 133 | 68 | 65 |
| 105 | Sublimation Print Tee | 2.4 | 3.3 | 89 | 46 | 43 |
| 106 | Brushed Cotton Trousers | 6.6 | 6.9 | 198 | 97 | 101 |
| 107 | Micro‑Waffle Knit Shirt | 3.1 | 4.2 | 125 | 63 | 62 |
| 108 | Reflective Detail Tee | 2.0 | 2.9 | 80 | 39 | 41 |
| 109 | Shearling Lined Coat | 7.9 | 8.2 | 238 | 138 | 100 |
| 110 | Printed Patchwork Shirt | 3.4 | 4.5 | 130 | 66 | 64 |
| 111 | Washed Effect T‑Shirt | 2.6 | 3.5 | 92 | 49 | 43 |
| 112 | Velour Track Pants | 5.8 | 6.0 | 182 | 86 | 96 |
| 113 | Fine Wale Corduroy Shirt | 3.7 | 4.8 | 138 | 72 | 66 |
| 114 | Organic Hemp T‑Shirt | 2.8 | 3.7 | 96 | 52 | 44 |
| 115 | Leather Biker Jacket | 8.2 | 8.5 | 250 | 150 | 100 |
| 116 | Supersoft Modal Shirt | 3.0 | 4.1 | 123 | 61 | 62 |
| 117 | Deep V‑Neck Tee | 2.3 | 3.2 | 87 | 44 | 43 |
| 118 | Wool Blend Trousers | 6.8 | 7.1 | 203 | 103 | 100 |
| 119 | Contrast Stitch Shirt | 3.9 | 5.0 | 142 | 75 | 67 |
| 120 | Embroidered Pocket Tee | 2.5 | 3.4 | 90 | 47 | 43 |
| 121 | Camo Print Hoodie | 6.1 | 6.3 | 187 | 89 | 98 |
| 122 | Silk‑Touch Poplin Shirt | 3.3 | 4.4 | 129 | 66 | 63 |
| 123 | Slim Fit Pique Polo | 2.7 | 3.6 | 94 | 50 | 44 |
| 124 | Technical Cargo Vest | 6.9 | 7.2 | 206 | 106 | 100 |
| 125 | Cutaway Collar Shirt | 3.6 | 4.7 | 136 | 70 | 66 |
| 126 | Lycra Blend Active Tee | 2.1 | 3.0 | 81 | 40 | 41 |
| 127 | Heavy Knit Sweater | 7.3 | 7.6 | 218 | 117 | 101 |
| 128 | Textured Stripe Shirt | 3.2 | 4.3 | 127 | 64 | 63 |
| 129 | Bio‑Washed T‑Shirt | 2.8 | 3.7 | 97 | 53 | 44 |
| 130 | Mohair Blend Cardigan | 7.0 | 7.3 | 211 | 110 | 101 |
| 131 | Shantung Silk Shirt | 4.1 | 5.2 | 148 | 80 | 68 |
| 132 | Distressed Denim Tee | 2.0 | 2.9 | 79 | 38 | 41 |
| 133 | High‑Tech Ski Jacket | 8.5 | 8.8 | 260 | 160 | 100 |
| 134 | Fine Poplin Check Shirt | 3.5 | 4.6 | 133 | 68 | 65 |
| 135 | Slim Fit Linen T‑Shirt | 2.2 | 3.1 | 85 | 43 | 42 |
| 136 | Ripstop Utility Pants | 6.3 | 6.5 | 190 | 91 | 99 |
| 137 | Classic Barrel Cuff Shirt | 3.7 | 4.8 | 139 | 73 | 66 |
| 138 | Bamboo Fiber Tee | 2.4 | 3.3 | 89 | 46 | 43 |
| 139 | Shearling Trench Coat | 8.1 | 8.4 | 245 | 145 | 100 |
| 140 | Slim Fit Chambray | 3.0 | 4.1 | 124 | 62 | 62 |
| 141 | Printed Geometric Tee | 2.5 | 3.4 | 91 | 48 | 43 |
| 142 | Wool Cargo Trousers | 6.7 | 7.0 | 201 | 101 | 100 |
| 143 | Two‑Tone Oxford Shirt | 3.3 | 4.4 | 130 | 67 | 63 |
| 144 | Slogan Print T‑Shirt | 2.6 | 3.5 | 93 | 50 | 43 |
| 145 | Water‑Resistant Shell | 7.1 | 7.4 | 213 | 113 | 100 |
| 146 | Grandad Collar Shirt | 3.8 | 4.9 | 140 | 74 | 66 |
| 147 | Embroidered Logo Tee | 2.3 | 3.2 | 87 | 45 | 42 |
| 148 | Casual Knit Blazer | 6.5 | 6.8 | 196 | 96 | 100 |
| 149 | Eco‑Friendly Hemp Shirt | 3.1 | 4.2 | 126 | 63 | 63 |
| 150 | Cotton Slub Polo | 2.9 | 3.8 | 98 | 54 | 44 |
| 151 | Corduroy Carpenter Pants | 6.2 | 6.4 | 188 | 90 | 98 |
| 152 | Linen Safari Shirt | 3.4 | 4.5 | 131 | 68 | 63 |
| 153 | Ribbed Crew Neck Tee | 2.7 | 3.6 | 95 | 51 | 44 |
| 154 | Hybrid Hiking Trousers | 6.0 | 6.2 | 185 | 87 | 98 |
| 155 | Brushed Flannel Shirt | 3.6 | 4.7 | 135 | 70 | 65 |
| 156 | Microfiber V‑Neck | 2.0 | 2.9 | 78 | 37 | 41 |
| 157 | Down‑Alternative Vest | 6.9 | 7.2 | 205 | 104 | 101 |
| 158 | Checked Seersucker Shirt | 3.9 | 5.0 | 141 | 75 | 66 |
| 159 | Fitted Basic Tee | 2.4 | 3.3 | 88 | 46 | 42 |
| 160 | Heavyweight Denim Jacket | 7.4 | 7.7 | 220 | 119 | 101 |
| 161 | Woven Stripe Shirt | 3.2 | 4.3 | 128 | 65 | 63 |
| 162 | Tech Fleece Pullover | 6.6 | 6.9 | 196 | 95 | 101 |
| 163 | Pin Dot Print Shirt | 3.5 | 4.6 | 133 | 69 | 64 |
| 164 | Supersoft Modal T‑Shirt | 2.5 | 3.4 | 90 | 47 | 43 |
| 165 | Water‑Resistant Chinos | 6.4 | 6.6 | 192 | 93 | 99 |
| 166 | Tailored Fit Dress Shirt | 3.7 | 4.8 | 138 | 72 | 66 |
| 167 | Printed Art Graphic Tee | 2.2 | 3.1 | 86 | 43 | 43 |
| 168 | Cashmere Cable Knit | 7.6 | 7.9 | 230 | 129 | 101 |
| 169 | Plaid Flannel Overshirt | 3.0 | 4.1 | 124 | 62 | 62 |
| 170 | Moisture‑Wicking Polo | 2.3 | 3.2 | 87 | 45 | 42 |
| 171 | Slim Tapered Jeans | 6.8 | 7.1 | 203 | 102 | 101 |
| 172 | Contrast Cuff Shirt | 3.3 | 4.4 | 129 | 66 | 63 |
| 173 | Sustainable Cotton Tee | 2.6 | 3.5 | 93 | 50 | 43 |
| 174 | Trench Coat (Lined) | 8.3 | 8.6 | 255 | 155 | 100 |
| 175 | Spread Collar Shirt | 3.8 | 4.9 | 140 | 73 | 67 |
| 176 | Printed Stripe Tee | 2.7 | 3.6 | 97 | 53 | 44 |
| 177 | Non‑Iron Twill Shirt | 3.1 | 4.2 | 126 | 64 | 62 |
| 178 | Recycled Material Tee | 2.0 | 2.9 | 79 | 38 | 41 |
| 179 | Faux Fur Coat | 7.7 | 8.0 | 235 | 134 | 101 |
| 180 | Seersucker Popover Shirt | 3.4 | 4.5 | 131 | 67 | 64 |
| 181 | Graphic Back Print Tee | 2.4 | 3.3 | 89 | 46 | 43 |
| 182 | Corduroy Field Jacket | 6.7 | 7.0 | 200 | 98 | 102 |
| 183 | Linen/Cotton Blend Shirt | 3.6 | 4.7 | 136 | 71 | 65 |
| 184 | Lightweight Jersey Tee | 2.5 | 3.4 | 91 | 48 | 43 |
| 185 | High Performance Hiking Pants | 6.5 | 6.8 | 194 | 94 | 100 |
| 186 | Cuban Collar Shirt | 3.9 | 5.0 | 142 | 76 | 66 |
| 187 | Faded Pigment Tee | 2.8 | 3.7 | 98 | 54 | 44 |
| 188 | Canvas Chore Coat | 7.2 | 7.5 | 216 | 116 | 100 |
| 189 | Brushed Cotton Shirt | 3.2 | 4.3 | 127 | 65 | 62 |
| 190 | Ribbed Knit T‑Shirt | 2.1 | 3.0 | 83 | 41 | 42 |
| 191 | Wide Leg Trousers | 6.1 | 6.3 | 187 | 89 | 98 |
| 192 | Patchwork Denim Shirt | 3.5 | 4.6 | 133 | 69 | 64 |
| 193 | Basic White T‑Shirt | 2.3 | 3.2 | 87 | 44 | 43 |
| 194 | Mohair Knit Sweater | 8.0 | 8.3 | 224 | 123 | 101 |
| 195 | Contrast Stitch Polo | 2.9 | 3.8 | 99 | 55 | 44 |
| 196 | Tailored Wool Trousers | 6.6 | 6.9 | 200 | 99 | 101 |
| 197 | Heavyweight Denim Shirt | 4.0 | 5.1 | 145 | 77 | 68 |

The model above fully specifies the optimization problem using the real data retrieved from the provided dataset.