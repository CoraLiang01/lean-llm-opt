##### Decision Variables  

$x_{p}\geq0$: quantity produced of product $p$ (continuous).  

##### Objective Function  

$\displaystyle \max \Bigl(\sum_{p\in P}\bigl(\text{SellingPrice}_{p}-\text{VariableCost}_{p}\bigr)\,x_{p}\;-\;4500\Bigr)$  

##### Constraints  

1. **Labor capacity**  
$\displaystyle \sum_{p\in P}\text{LaborPerUnit}_{p}\,x_{p}\;\leq\;1650$  

2. **Material capacity**  
$\displaystyle \sum_{p\in P}\text{MaterialPerUnit}_{p}\,x_{p}\;\leq\;1850$  

3. **Non‑negativity**  
$\displaystyle x_{p}\;\geq\;0,\qquad \forall p\in P$  

##### Data (products in the order given in the CSV)

| Product | Labor per unit | Material per unit | Selling Price | Variable Cost | Profit per unit |
|---------|----------------|-------------------|---------------|---------------|-----------------|
| Classic Oxford Shirt | 3.1 | 4.2 | 125 | 63 | 62 |
| Cotton Crew Neck T-Shirt | 2.1 | 3.1 | 83 | 41 | 42 |
| French Terry Hoodie | 6.5 | 6.8 | 195 | 95 | 100 |
| Denim Snap Shirt | 3.4 | 4.6 | 132 | 68 | 64 |
| Jersey V-Neck T-Shirt | 2.5 | 3.4 | 90 | 48 | 42 |
| Unisex Jogger Pants | 5.9 | 6.1 | 185 | 88 | 97 |
| Flannel Plaid Shirt | 3.5 | 4.0 | 128 | 65 | 63 |
| Performance Polo Shirt | 2.9 | 3.9 | 118 | 57 | 61 |
| Long-sleeve Henley | 2.8 | 3.7 | 95 | 50 | 45 |
| Heavyweight Sweatshirt | 6.2 | 6.4 | 190 | 92 | 98 |
| Linen Blend Shirt | 3.2 | 4.3 | 127 | 64 | 63 |
| Graphic Print Tee | 2.0 | 3.0 | 82 | 40 | 42 |
| Quilted Vest | 6.8 | 7.0 | 205 | 105 | 100 |
| Chambray Work Shirt | 3.0 | 4.1 | 122 | 61 | 61 |
| Microfiber Sport Shirt | 2.3 | 3.3 | 88 | 45 | 43 |
| Wool Blend Peacoat | 7.2 | 7.5 | 215 | 115 | 100 |
| Seersucker Short-sleeve | 2.7 | 3.6 | 93 | 47 | 46 |
| Twill Utility Shirt | 3.3 | 4.5 | 130 | 67 | 63 |
| Basic Scoop Neck Tee | 2.2 | 3.2 | 84 | 42 | 42 |
| Waffle Knit Pullover | 6.1 | 6.3 | 187 | 91 | 96 |
| Button-Down Poplin Shirt | 3.6 | 4.7 | 135 | 70 | 65 |
| Slub Cotton T-Shirt | 2.6 | 3.5 | 92 | 49 | 43 |
| Drawstring Cargo Pants | 6.4 | 6.6 | 192 | 93 | 99 |
| Pima Cotton Dress Shirt | 3.8 | 4.9 | 140 | 73 | 67 |
| Pocket T-Shirt | 2.4 | 3.3 | 87 | 44 | 43 |
| Fleece-Lined Track Suit | 7.0 | 7.2 | 210 | 110 | 100 |
| Printed Camp Collar Shirt | 2.9 | 4.0 | 120 | 60 | 60 |
| Performance Dry-Fit Tee | 2.0 | 3.0 | 80 | 39 | 41 |
| Corduroy Shacket | 6.7 | 6.9 | 202 | 97 | 105 |
| Herringbone Casual Shirt | 3.0 | 4.1 | 123 | 62 | 61 |
| Organic Cotton Tee | 2.1 | 3.1 | 85 | 43 | 42 |
| Cashmere Blend Sweater | 7.5 | 7.8 | 225 | 125 | 100 |
| Striped Rugby Shirt | 3.4 | 4.4 | 129 | 66 | 63 |
| Raglan Sleeve Tee | 2.6 | 3.6 | 94 | 51 | 43 |
| Waterproof Rain Jacket | 7.1 | 7.4 | 212 | 112 | 100 |
| Micro-check Business Shirt | 3.7 | 4.8 | 138 | 72 | 66 |
| Heather Grey T-Shirt | 2.3 | 3.2 | 86 | 45 | 41 |
| Slim Fit Chinos | 6.3 | 6.5 | 190 | 90 | 100 |
| Band Collar Shirt | 3.1 | 4.2 | 126 | 64 | 62 |
| Triblend Jersey Tee | 2.5 | 3.5 | 91 | 49 | 42 |
| Sherpa Lined Pullover | 6.9 | 7.1 | 208 | 108 | 100 |
| Western Style Shirt | 3.2 | 4.3 | 128 | 65 | 63 |
| Recycled Poly T-Shirt | 2.7 | 3.7 | 96 | 52 | 44 |
| Insulated Puffer Jacket | 7.4 | 7.7 | 220 | 120 | 100 |
| End-on-End Fabric Shirt | 3.5 | 4.6 | 134 | 69 | 65 |
| Oversized Graphic Tee | 2.8 | 3.8 | 98 | 54 | 44 |
| Stretch Hybrid Shorts | 6.0 | 6.2 | 186 | 89 | 97 |
| Non-Iron Travel Shirt | 3.9 | 5.0 | 142 | 75 | 67 |
| Sun-Protective T-Shirt | 2.1 | 3.0 | 81 | 40 | 41 |
| Velvet Bomber Jacket | 7.3 | 7.6 | 218 | 118 | 100 |
| Madras Plaid Shirt | 3.3 | 4.4 | 129 | 66 | 63 |
| Tie-Dye T-Shirt | 2.4 | 3.3 | 89 | 46 | 43 |
| Lightweight Windbreaker | 5.8 | 6.0 | 180 | 85 | 95 |
| Pinpoint Oxford Shirt | 3.6 | 4.7 | 136 | 71 | 65 |
| Bamboo Viscose Tee | 2.0 | 2.9 | 79 | 38 | 41 |
| Tapered Denim Jeans | 6.6 | 6.9 | 198 | 96 | 102 |
| Contrast Collar Shirt | 3.8 | 4.9 | 141 | 74 | 67 |
| Mock Neck Long-sleeve | 2.2 | 3.1 | 86 | 43 | 43 |
| Down-filled Parka | 7.8 | 8.1 | 235 | 135 | 100 |
| Textured Grid Shirt | 3.1 | 4.2 | 125 | 63 | 62 |
| Heavy Cotton T-Shirt | 2.5 | 3.4 | 90 | 48 | 42 |
| Convertible Zip-Off Trousers | 6.5 | 6.8 | 195 | 94 | 101 |
| Tencel Blend Casual Shirt | 3.4 | 4.5 | 131 | 67 | 64 |
| Relaxed Fit Tee | 2.7 | 3.6 | 95 | 51 | 44 |
| Faux Leather Jacket | 7.0 | 7.3 | 210 | 111 | 99 |
| Speckled Slub Shirt | 3.0 | 4.1 | 124 | 62 | 62 |
| Embroidered T-Shirt | 2.3 | 3.2 | 88 | 45 | 43 |
| Knit Cardigan Sweater | 6.1 | 6.3 | 188 | 90 | 98 |
| Two-Pocket Utility Shirt | 3.7 | 4.8 | 139 | 72 | 67 |
| Performance Stretch Tee | 2.6 | 3.5 | 93 | 50 | 43 |
| Technical Shell Jacket | 7.6 | 7.9 | 230 | 130 | 100 |
| Semi-Spread Collar Shirt | 3.2 | 4.3 | 127 | 64 | 63 |
| Washed Pigment Dye Tee | 2.1 | 3.0 | 82 | 40 | 42 |
| Twill Blazer (Unlined) | 6.9 | 7.1 | 205 | 100 | 105 |
| Brushed Twill Shirt | 3.5 | 4.6 | 134 | 69 | 65 |
| Supima Cotton Tee | 2.8 | 3.7 | 97 | 53 | 44 |
| High-Waist Trousers | 6.2 | 6.4 | 189 | 91 | 98 |
| Printed Geometric Shirt | 3.9 | 5.0 | 143 | 76 | 67 |
| Minimalist Plain Tee | 2.0 | 2.9 | 78 | 37 | 41 |
| Reversible Quilted Jacket | 7.4 | 7.7 | 222 | 122 | 100 |
| Dobby Weave Shirt | 3.3 | 4.4 | 130 | 67 | 63 |
| Heavyweight Boxy Tee | 2.4 | 3.3 | 89 | 46 | 43 |
| Casual Wool Overcoat | 8.0 | 8.3 | 240 | 140 | 100 |
| Half-Zip Pullover Shirt | 3.6 | 4.7 | 137 | 71 | 66 |
| Drop Shoulder T-Shirt | 2.2 | 3.1 | 87 | 44 | 43 |
| Straight Leg Corduroys | 6.7 | 7.0 | 200 | 99 | 101 |
| Camp Collar Floral Shirt | 2.9 | 4.0 | 121 | 61 | 60 |
| Garment Dyed Tee | 2.5 | 3.4 | 90 | 47 | 43 |
| Suede Bomber Jacket | 7.7 | 8.0 | 232 | 132 | 100 |
| Melange Knit Shirt | 3.1 | 4.2 | 126 | 64 | 62 |
| Longline T-Shirt | 2.6 | 3.5 | 94 | 51 | 43 |
| Wide Leg Culottes | 6.3 | 6.5 | 191 | 92 | 99 |
| Tropical Print Shirt | 3.8 | 4.9 | 140 | 73 | 67 |
| Distressed Cotton Tee | 2.3 | 3.2 | 88 | 45 | 43 |
| Patchwork Denim Jacket | 7.2 | 7.5 | 215 | 114 | 101 |
| Micro-Dot Shirt | 3.0 | 4.1 | 123 | 62 | 61 |
| Muscle Fit T-Shirt | 2.1 | 3.0 | 83 | 41 | 42 |
| Woven Utility Vest | 6.4 | 6.6 | 193 | 93 | 100 |
| Silk Blend Dress Shirt | 4.0 | 5.1 | 145 | 78 | 67 |
| Heavyweight Jersey T-Shirt | 2.7 | 3.6 | 96 | 52 | 44 |
| Slim Fit Ankle Grazer | 6.0 | 6.2 | 186 | 88 | 98 |
| Lattice Weave Shirt | 3.2 | 4.3 | 128 | 65 | 63 |
| Boxy Cropped Tee | 2.2 | 3.1 | 84 | 42 | 42 |
| Reversible Bomber Jacket | 7.5 | 7.8 | 225 | 124 | 101 |
| Stretch Performance Shirt | 3.5 | 4.6 | 133 | 68 | 65 |
| Sublimation Print Tee | 2.4 | 3.3 | 89 | 46 | 43 |
| Brushed Cotton Trousers | 6.6 | 6.9 | 198 | 97 | 101 |
| Micro-Waffle Knit Shirt | 3.1 | 4.2 | 125 | 63 | 62 |
| Reflective Detail Tee | 2.0 | 2.9 | 80 | 39 | 41 |
| Shearling Lined Coat | 7.9 | 8.2 | 238 | 138 | 100 |
| Printed Patchwork Shirt | 3.4 | 4.5 | 130 | 66 | 64 |
| Washed Effect T-Shirt | 2.6 | 3.5 | 92 | 49 | 43 |
| Velour Track Pants | 5.8 | 6.0 | 182 | 86 | 96 |
| Fine Wale Corduroy Shirt | 3.7 | 4.8 | 138 | 72 | 66 |
| Organic Hemp T-Shirt | 2.8 | 3.7 | 96 | 52 | 44 |
| Leather Biker Jacket | 8.2 | 8.5 | 250 | 150 | 100 |
| Supersoft Modal Shirt | 3.0 | 4.1 | 123 | 61 | 62 |
| Deep V-Neck Tee | 2.3 | 3.2 | 87 | 44 | 43 |
| Wool Blend Trousers | 6.8 | 7.1 | 203 | 103 | 100 |
| Contrast Stitch Shirt | 3.9 | 5.0 | 142 | 75 | 67 |
| Embroidered Pocket Tee | 2.5 | 3.4 | 90 | 47 | 43 |
| Camo Print Hoodie | 6.1 | 6.3 | 187 | 89 | 98 |
| Silk-Touch Poplin Shirt | 3.3 | 4.4 | 129 | 66 | 63 |
| Slim Fit Pique Polo | 2.7 | 3.6 | 94 | 50 | 44 |
| Technical Cargo Vest | 6.9 | 7.2 | 206 | 106 | 100 |
| Cutaway Collar Shirt | 3.6 | 4.7 | 136 | 70 | 66 |
| Lycra Blend Active Tee | 2.1 | 3.0 | 81 | 40 | 41 |
| Heavy Knit Sweater | 7.3 | 7.6 | 218 | 117 | 101 |
| Textured Stripe Shirt | 3.2 | 4.3 | 127 | 64 | 63 |
| Bio-Washed T-Shirt | 2.8 | 3.7 | 97 | 53 | 44 |
| Mohair Blend Cardigan | 7.0 | 7.3 | 211 | 110 | 101 |
| Shantung Silk Shirt | 4.1 | 5.2 | 148 | 80 | 68 |
| Distressed Denim Tee | 2.0 | 2.9 | 79 | 38 | 41 |
| High-Tech Ski Jacket | 8.5 | 8.8 | 260 | 160 | 100 |
| Fine Poplin Check Shirt | 3.5 | 4.6 | 133 | 68 | 65 |
| Slim Fit Linen T-Shirt | 2.2 | 3.1 | 85 | 43 | 42 |
| Ripstop Utility Pants | 6.3 | 6.5 | 190 | 91 | 99 |
| Classic Barrel Cuff Shirt | 3.7 | 4.8 | 139 | 73 | 66 |
| Bamboo Fiber Tee | 2.4 | 3.3 | 89 | 46 | 43 |
| Shearling Trench Coat | 8.1 | 8.4 | 245 | 145 | 100 |
| Slim Fit Chambray | 3.0 | 4.1 | 124 | 62 | 62 |
| Printed Geometric Tee | 2.5 | 3.4 | 91 | 48 | 43 |
| Wool Cargo Trousers | 6.7 | 7.0 | 201 | 101 | 100 |
| Two-Tone Oxford Shirt | 3.3 | 4.4 | 130 | 67 | 63 |
| Slogan Print T-Shirt | 2.6 | 3.5 | 93 | 50 | 43 |
| Water-Resistant Shell | 7.1 | 7.4 | 213 | 113 | 100 |
| Grandad Collar Shirt | 3.8 | 4.9 | 140 | 74 | 66 |
| Embroidered Logo Tee | 2.3 | 3.2 | 87 | 45 | 42 |
| Casual Knit Blazer | 6.5 | 6.8 | 196 | 96 | 100 |
| Eco-Friendly Hemp Shirt | 3.1 | 4.2 | 126 | 63 | 63 |
| Cotton Slub Polo | 2.9 | 3.8 | 98 | 54 | 44 |
| Corduroy Carpenter Pants | 6.2 | 6.4 | 188 | 90 | 98 |
| Linen Safari Shirt | 3.4 | 4.5 | 131 | 68 | 63 |
| Ribbed Crew Neck Tee | 2.7 | 3.6 | 95 | 51 | 44 |
| Hybrid Hiking Trousers | 6.0 | 6.2 | 185 | 87 | 98 |
| Brushed Flannel Shirt | 3.6 | 4.7 | 135 | 70 | 65 |
| Microfiber V-Neck | 2.0 | 2.9 | 78 | 37 | 41 |
| Down-Alternative Vest | 6.9 | 7.2 | 205 | 104 | 101 |
| Checked Seersucker Shirt | 3.9 | 5.0 | 141 | 75 | 66 |
| Fitted Basic Tee | 2.4 | 3.3 | 88 | 46 | 42 |
| Heavyweight Denim Jacket | 7.4 | 7.7 | 220 | 119 | 101 |
| Woven Stripe Shirt | 3.2 | 4.3 | 128 | 65 | 63 |
| Tech Fleece Pullover | 6.6 | 6.9 | 196 | 95 | 101 |
| Pin Dot Print Shirt | 3.5 | 4.6 | 133 | 69 | 64 |
| Supersoft Modal T-Shirt | 2.5 | 3.4 | 90 | 47 | 43 |
| Water-Resistant Chinos | 6.4 | 6.6 | 192 | 93 | 99 |
| Tailored Fit Dress Shirt | 3.7 | 4.8 | 138 | 72 | 66 |
| Printed Art Graphic Tee | 2.2 | 3.1 | 86 | 43 | 43 |
| Cashmere Cable Knit | 7.6 | 7.9 | 230 | 129 | 101 |
| Plaid Flannel Overshirt | 3.0 | 4.1 | 124 | 62 | 62 |
| Moisture-Wicking Polo | 2.3 | 3.2 | 87 | 45 | 42 |
| Slim Tapered Jeans | 6.8 | 7.1 | 203 | 102 | 101 |
| Contrast Cuff Shirt | 3.3 | 4.4 | 129 | 66 | 63 |
| Sustainable Cotton Tee | 2.6 | 3.5 | 93 | 50 | 43 |
| Trench Coat (Lined) | 8.3 | 8.6 | 255 | 155 | 100 |
| Spread Collar Shirt | 3.8 | 4.9 | 140 | 73 | 67 |
| Printed Stripe Tee | 2.7 | 3.6 | 97 | 53 | 44 |
| Non-Iron Twill Shirt | 3.1 | 4.2 | 126 | 64 | 62 |
| Recycled Material Tee | 2.0 | 2.9 | 79 | 38 | 41 |
| Faux Fur Coat | 7.7 | 8.0 | 235 | 134 | 101 |
| Seersucker Popover Shirt | 3.4 | 4.5 | 131 | 67 | 64 |
| Graphic Back Print Tee | 2.4 | 3.3 | 89 | 46 | 43 |
| Corduroy Field Jacket | 6.7 | 7.0 | 200 | 98 | 102 |
| Linen/Cotton Blend Shirt | 3.6 | 4.7 | 136 | 71 | 65 |
| Lightweight Jersey Tee | 2.5 | 3.4 | 91 | 48 | 43 |
| High Performance Hiking Pants | 6.5 | 6.8 | 194 | 94 | 100 |
| Cuban Collar Shirt | 3.9 | 5.0 | 142 | 76 | 66 |
| Faded Pigment Tee | 2.8 | 3.7 | 98 | 54 | 44 |
| Canvas Chore Coat | 7.2 | 7.5 | 216 | 116 | 100 |
| Brushed Cotton Shirt | 3.2 | 4.3 | 127 | 65 | 62 |
| Ribbed Knit T-Shirt | 2.1 | 3.0 | 83 | 41 | 42 |
| Wide Leg Trousers | 6.1 | 6.3 | 187 | 89 | 98 |
| Patchwork Denim Shirt | 3.5 | 4.6 | 133 | 69 | 64 |
| Basic White T-Shirt | 2.3 | 3.2 | 87 | 44 | 43 |
| Mohair Knit Sweater | 7.5 | 7.8 | 224 | 123 | 101 |
| Contrast Stitch Polo | 6.6 | 6.9 | 199 | 99 | 100 |
| Tailored Wool Trousers | 6.6 | 6.9 | 200 | 99 | 101 |
| Heavyweight Denim Shirt | 4.0 | 5.1 | 145 | 77 | 68 |

All coefficients are taken directly from the CSV files; no additional data were omitted or aggregated.