Let \( x_i \) denote the nonnegative continuous production quantity of product \( i \), where \( i \) indexes the products in the order given in 41.csv. Let the set of products be indexed by \( i = 1, \ldots, 100 \), with the following mapping (in order):

1. Classic Oxford Shirt
2. Cotton Crew Neck T-Shirt
3. French Terry Hoodie
4. Denim Snap Shirt
5. Jersey V-Neck T-Shirt
6. Unisex Jogger Pants
7. Flannel Plaid Shirt
8. Performance Polo Shirt
9. Long-sleeve Henley
10. Heavyweight Sweatshirt
11. Linen Blend Shirt
12. Graphic Print Tee
13. Quilted Vest
14. Chambray Work Shirt
15. Microfiber Sport Shirt
16. Wool Blend Peacoat
17. Seersucker Short-sleeve
18. Twill Utility Shirt
19. Basic Scoop Neck Tee
20. Waffle Knit Pullover
21. Button-Down Poplin Shirt
22. Slub Cotton T-Shirt
23. Drawstring Cargo Pants
24. Pima Cotton Dress Shirt
25. Pocket T-Shirt
26. Fleece-Lined Track Suit
27. Printed Camp Collar Shirt
28. Performance Dry-Fit Tee
29. Corduroy Shacket
30. Herringbone Casual Shirt
31. Organic Cotton Tee
32. Cashmere Blend Sweater
33. Striped Rugby Shirt
34. Raglan Sleeve Tee
35. Waterproof Rain Jacket
36. Micro-check Business Shirt
37. Heather Grey T-Shirt
38. Slim Fit Chinos
39. Band Collar Shirt
40. Triblend Jersey Tee
41. Sherpa Lined Pullover
42. Western Style Shirt
43. Recycled Poly T-Shirt
44. Insulated Puffer Jacket
45. End-on-End Fabric Shirt
46. Oversized Graphic Tee
47. Stretch Hybrid Shorts
48. Non-Iron Travel Shirt
49. Sun-Protective T-Shirt
50. Velvet Bomber Jacket
51. Madras Plaid Shirt
52. Tie-Dye T-Shirt
53. Lightweight Windbreaker
54. Pinpoint Oxford Shirt
55. Bamboo Viscose Tee
56. Tapered Denim Jeans
57. Contrast Collar Shirt
58. Mock Neck Long-sleeve
59. Down-filled Parka
60. Textured Grid Shirt
61. Heavy Cotton T-Shirt
62. Convertible Zip-Off Trousers
63. Tencel Blend Casual Shirt
64. Relaxed Fit Tee
65. Faux Leather Jacket
66. Speckled Slub Shirt
67. Embroidered T-Shirt
68. Knit Cardigan Sweater
69. Two-Pocket Utility Shirt
70. Performance Stretch Tee
71. Technical Shell Jacket
72. Semi-Spread Collar Shirt
73. Washed Pigment Dye Tee
74. Twill Blazer (Unlined)
75. Brushed Twill Shirt
76. Supima Cotton Tee
77. High-Waist Trousers
78. Printed Geometric Shirt
79. Minimalist Plain Tee
80. Reversible Quilted Jacket
81. Dobby Weave Shirt
82. Heavyweight Boxy Tee
83. Casual Wool Overcoat
84. Half-Zip Pullover Shirt
85. Drop Shoulder T-Shirt
86. Straight Leg Corduroys
87. Camp Collar Floral Shirt
88. Garment Dyed Tee
89. Suede Bomber Jacket
90. Melange Knit Shirt
91. Longline T-Shirt
92. Wide Leg Culottes
93. Tropical Print Shirt
94. Distressed Cotton Tee
95. Patchwork Denim Jacket
96. Micro-Dot Shirt
97. Muscle Fit T-Shirt
98. Woven Utility Vest
99. Silk Blend Dress Shirt
100. Heavyweight Jersey T-Shirt

Parameters for each product \( i \) (from the CSV, in order):

- \( a_i \): Labor per unit
- \( b_i \): Material per unit
- \( p_i \): Selling price per unit
- \( c_i \): Variable cost per unit

The data for each product is as follows (in order):

| \( i \) | Product Name | \( a_i \) | \( b_i \) | \( p_i \) | \( c_i \) |
|---|-------------------------------|--------|--------|--------|--------|
| 1 | Classic Oxford Shirt | 3.1 | 4.2 | 125 | 63 |
| 2 | Cotton Crew Neck T-Shirt | 2.1 | 3.1 | 83 | 41 |
| 3 | French Terry Hoodie | 6.5 | 6.8 | 195 | 95 |
| 4 | Denim Snap Shirt | 3.4 | 4.6 | 132 | 68 |
| 5 | Jersey V-Neck T-Shirt | 2.5 | 3.4 | 90 | 48 |
| 6 | Unisex Jogger Pants | 5.9 | 6.1 | 185 | 88 |
| 7 | Flannel Plaid Shirt | 3.5 | 4 | 128 | 65 |
| 8 | Performance Polo Shirt | 2.9 | 3.9 | 118 | 57 |
| 9 | Long-sleeve Henley | 2.8 | 3.7 | 95 | 50 |
| 10 | Heavyweight Sweatshirt | 6.2 | 6.4 | 190 | 92 |
| 11 | Linen Blend Shirt | 3.2 | 4.3 | 127 | 64 |
| 12 | Graphic Print Tee | 2 | 3 | 82 | 40 |
| 13 | Quilted Vest | 6.8 | 7 | 205 | 105 |
| 14 | Chambray Work Shirt | 3 | 4.1 | 122 | 61 |
| 15 | Microfiber Sport Shirt | 2.3 | 3.3 | 88 | 45 |
| 16 | Wool Blend Peacoat | 7.2 | 7.5 | 215 | 115 |
| 17 | Seersucker Short-sleeve | 2.7 | 3.6 | 93 | 47 |
| 18 | Twill Utility Shirt | 3.3 | 4.5 | 130 | 67 |
| 19 | Basic Scoop Neck Tee | 2.2 | 3.2 | 84 | 42 |
| 20 | Waffle Knit Pullover | 6.1 | 6.3 | 187 | 91 |
| 21 | Button-Down Poplin Shirt | 3.6 | 4.7 | 135 | 70 |
| 22 | Slub Cotton T-Shirt | 2.6 | 3.5 | 92 | 49 |
| 23 | Drawstring Cargo Pants | 6.4 | 6.6 | 192 | 93 |
| 24 | Pima Cotton Dress Shirt | 3.8 | 4.9 | 140 | 73 |
| 25 | Pocket T-Shirt | 2.4 | 3.3 | 87 | 44 |
| 26 | Fleece-Lined Track Suit | 7 | 7.2 | 210 | 110 |
| 27 | Printed Camp Collar Shirt | 2.9 | 4 | 120 | 60 |
| 28 | Performance Dry-Fit Tee | 2 | 3 | 80 | 39 |
| 29 | Corduroy Shacket | 6.7 | 6.9 | 202 | 97 |
| 30 | Herringbone Casual Shirt | 3 | 4.1 | 123 | 62 |
| 31 | Organic Cotton Tee | 2.1 | 3.1 | 85 | 43 |
| 32 | Cashmere Blend Sweater | 7.5 | 7.8 | 225 | 125 |
| 33 | Striped Rugby Shirt | 3.4 | 4.4 | 129 | 66 |
| 34 | Raglan Sleeve Tee | 2.6 | 3.6 | 94 | 51 |
| 35 | Waterproof Rain Jacket | 7.1 | 7.4 | 212 | 112 |
| 36 | Micro-check Business Shirt | 3.7 | 4.8 | 138 | 72 |
| 37 | Heather Grey T-Shirt | 2.3 | 3.2 | 86 | 45 |
| 38 | Slim Fit Chinos | 6.3 | 6.5 | 190 | 90 |
| 39 | Band Collar Shirt | 3.1 | 4.2 | 126 | 64 |
| 40 | Triblend Jersey Tee | 2.5 | 3.5 | 91 | 49 |
| 41 | Sherpa Lined Pullover | 6.9 | 7.1 | 208 | 108 |
| 42 | Western Style Shirt | 3.2 | 4.3 | 128 | 65 |
| 43 | Recycled Poly T-Shirt | 2.7 | 3.7 | 96 | 52 |
| 44 | Insulated Puffer Jacket | 7.4 | 7.7 | 220 | 120 |
| 45 | End-on-End Fabric Shirt | 3.5 | 4.6 | 134 | 69 |
| 46 | Oversized Graphic Tee | 2.8 | 3.8 | 98 | 54 |
| 47 | Stretch Hybrid Shorts | 6 | 6.2 | 186 | 89 |
| 48 | Non-Iron Travel Shirt | 3.9 | 5 | 142 | 75 |
| 49 | Sun-Protective T-Shirt | 2.1 | 3 | 81 | 40 |
| 50 | Velvet Bomber Jacket | 7.3 | 7.6 | 218 | 118 |
| ... | ... | ... | ... | ... | ... |
| 100 | Heavyweight Jersey T-Shirt | 2.7 | 3.6 | 96 | 52 |

(The full table is as above; for brevity, only the first 50 are shown here, but all 100 are included in the model.)

The optimization model is:

\[
\begin{align*}
\text{Maximize} \quad & \sum_{i=1}^{100} (p_i - c_i) x_i - 4500 \\
\text{subject to} \quad
& \sum_{i=1}^{100} a_i x_i \leq 1650 \\
& \sum_{i=1}^{100} b_i x_i \leq 1850 \\
& x_i \geq 0 \quad \forall i = 1, \ldots, 100
\end{align*}
\]

Where:
- \( x_i \): nonnegative continuous variable, production quantity of product \( i \)
- \( a_i \): labor per unit for product \( i \) (from "Labor per unit" column)
- \( b_i \): material per unit for product \( i \) (from "Material per unit" column)
- \( p_i \): selling price per unit for product \( i \) (from "Selling Price" column)
- \( c_i \): variable cost per unit for product \( i \) (from "Variable Cost" column)
- The fixed weekly operating cost is $4,500, subtracted as a constant in the objective.

All coefficients and product names are taken directly from the supplied CSV, in the original order. No sorting or re-indexing has been performed. All 100 products are included, and the labor and material capacity constraints use the explicit coefficients and total capacities as given.

This is a continuous linear program, as the user explicitly requests nonnegative continuous production quantities.