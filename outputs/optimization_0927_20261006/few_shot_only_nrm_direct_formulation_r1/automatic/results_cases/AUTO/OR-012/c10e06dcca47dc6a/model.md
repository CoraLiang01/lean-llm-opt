**Index Sets:**

Let  
$\mathcal{I}$ = set of all products in the dataset, indexed by $i$.

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$ = revenue per unit of product $i$ (from column "Revenue")
- $d_i$ = deterministic demand for product $i$ (from column "Demand")
- $I_i$ = initial inventory for product $i$ (from column "Initial Inventory")

**Variables:**  
For each $i \in \mathcal{I}$:
- $x_i$ = number of units of product $i$ to fulfill, integer, $x_i \geq 0$

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory and Demand Fulfillment Bounds:**  
For all $i \in \mathcal{I}$,
$$
0 \leq x_i \leq \min\{I_i, d_i\}
$$
or equivalently,
$$
x_i \leq I_i \\
x_i \leq d_i \\
x_i \geq 0 \\
x_i \in \mathbb{Z}
$$

---

**Retrieved Information (complete, in source order):**

| $i$ | Product Name | $A_i$ (Revenue) | $d_i$ (Demand) | $I_i$ (Initial Inventory) |
|-----|-----------------------------|----------------|----------------|-------------------------|
| 0 | 1984 by George Orwell | 19.99 | 6 | 20 |
| 1 | Adidas 3-Stripes Shorts | 29.99 | 7 | 30 |
| 2 | Adidas Essential Track Pants | 44.99 | 5 | 20 |
| 3 | Adidas FIFA World Cup Football | 29.99 | 4 | 20 |
| 4 | Adidas Originals Superstar Sneakers | 79.99 | 6 | 20 |
| 5 | Adidas Originals Trefoil Hoodie | 64.99 | 6 | 20 |
| 6 | Adidas Ultraboost Running Shoes | 179.99 | 3 | 10 |
| 7 | Adidas Ultraboost Shoes | 179.99 | 3 | 10 |
| 8 | Amazon Echo Dot (4th Gen) | 49.99 | 5 | 20 |
| 9 | Amazon Echo Show 10 | 249.99 | 2 | 10 |
| 10 | Amazon Fire TV Stick 4K | 49.99 | 5 | 20 |
| 11 | Anastasia Beverly Hills Brow Wiz | 23 | 3 | 10 |
| 12 | Anker PowerCore Portable Charger | 59.99 | 6 | 20 |
| 13 | Anova Precision Cooker | 199 | 3 | 10 |
| 14 | Anova Precision Oven | 599 | 2 | 10 |
| 15 | Apple AirPods Max | 549 | 2 | 10 |
| 16 | Apple AirPods Pro | 249.99 | 3 | 10 |
| 17 | Apple MacBook Air | 1199.99 | 2 | 10 |
| 18 | Apple MacBook Pro 16-inch | 2399 | 2 | 10 |
| 19 | Apple TV 4K | 179 | 3 | 10 |
| 20 | Apple Watch Series 8 | 399.99 | 4 | 20 |
| 21 | Apple iPad Air | 599.99 | 3 | 10 |
| 22 | Atomic Habits by James Clear | 16.99 | 6 | 20 |
| 23 | Babolat Pure Drive Tennis Racket | 199.99 | 5 | 20 |
| 24 | Becoming by Michelle Obama | 32.5 | 6 | 20 |
| 25 | Biore UV Aqua Rich Watery Essence Sunscreen | 15 | 2 | 10 |
| 26 | Blueair Classic 480i | 599.99 | 3 | 10 |
| 27 | Bose QuietComfort 35 Headphones | 299.99 | 2 | 10 |
| 28 | Bose QuietComfort 35 II Wireless Headphones | 299 | 2 | 10 |
| 29 | Bose SoundLink Color Bluetooth Speaker II | 129 | 2 | 10 |
| 30 | Bose SoundLink Revolve+ Speaker | 299.99 | 5 | 20 |
| 31 | Bose SoundSport Wireless Earbuds | 149.99 | 3 | 10 |
| 32 | Bowflex SelectTech 1090 Adjustable Dumbbells | 699.99 | 2 | 10 |
| 33 | Bowflex SelectTech 552 Dumbbells | 399.99 | 2 | 10 |
| 34 | Breville Nespresso Creatista Plus | 499.95 | 2 | 10 |
| 35 | Breville Smart Coffee Grinder Pro | 199.95 | 2 | 10 |
| 36 | Breville Smart Grill | 299.95 | 3 | 10 |
| 37 | Breville Smart Oven | 299.99 | 2 | 10 |
| 38 | Calvin Klein Boxer Briefs | 29.99 | 7 | 30 |
| 39 | Canon EOS R5 Camera | 3899.99 | 2 | 10 |
| 40 | Canon EOS Rebel T7i DSLR Camera | 749.99 | 2 | 10 |
| 41 | Caudalie Vinoperfect Radiance Serum | 79 | 2 | 10 |
| 42 | CeraVe Hydrating Facial Cleanser | 14.99 | 3 | 10 |
| 43 | Champion Reverse Weave Hoodie | 49.99 | 5 | 20 |
| 44 | Chanel No. 5 Perfume | 129.99 | 2 | 10 |
| 45 | Charlotte Tilbury Magic Cream | 100 | 2 | 10 |
| 46 | Clinique Dramatically Different Moisturizing Lotion | 29.5 | 2 | 10 |
| 47 | Clinique Moisture Surge | 52 | 2 | 10 |
| 48 | Columbia Fleece Jacket | 59.99 | 6 | 20 |
| 49 | Crock-Pot 6-Quart Slow Cooker | 49.99 | 3 | 10 |
| 50 | Cuisinart Coffee Center | 199.95 | 3 | 10 |
| 51 | Cuisinart Custom 14-Cup Food Processor | 199.99 | 2 | 10 |
| 52 | Cuisinart Griddler Deluxe | 159.99 | 2 | 10 |
| 53 | De'Longhi Magnifica Espresso Machine | 899.99 | 2 | 10 |
| 54 | Dr. Jart+ Cicapair Tiger Grass Color Correcting Treatment | 52 | 2 | 10 |
| 55 | Drunk Elephant C-Firma Day Serum | 78 | 2 | 10 |
| 56 | Dune by Frank Herbert | 25.99 | 6 | 20 |
| 57 | Dyson Pure Cool Link | 499.99 | 2 | 10 |
| 58 | Dyson Supersonic Hair Dryer | 399.99 | 5 | 20 |
| 59 | Dyson V11 Vacuum | 499.99 | 2 | 10 |
| 60 | Dyson V8 Absolute | 399.99 | 2 | 10 |
| 61 | Educated by Tara Westover | 28 | 4 | 20 |
| 62 | Estee Lauder Advanced Night Repair | 105 | 2 | 10 |
| 63 | Eufy RoboVac 11S | 219.99 | 5 | 20 |
| 64 | Fenty Beauty Killawatt Highlighter | 36 | 2 | 10 |
| 65 | First Aid Beauty Ultra Repair Cream | 34 | 3 | 10 |
| 66 | Fitbit Charge 5 | 129.99 | 3 | 10 |
| 67 | Fitbit Inspire 2 | 99.95 | 3 | 10 |
| 68 | Fitbit Luxe | 149.95 | 3 | 10 |
| 69 | Fitbit Versa 3 | 229.95 | 5 | 20 |
| 70 | Forever 21 Graphic Tee | 12.99 | 7 | 30 |
| 71 | Fresh Sugar Lip Treatment | 24 | 2 | 10 |
| 72 | Gap 1969 Original Fit Jeans | 59.99 | 5 | 20 |
| 73 | Gap Crewneck Sweatshirt | 34.99 | 6 | 20 |
| 74 | Gap Essential Crewneck T-Shirt | 19.99 | 8 | 30 |
| 75 | Gap High Rise Skinny Jeans | 49.99 | 5 | 20 |
| 76 | Garmin Edge 530 | 299.99 | 3 | 10 |
| 77 | Garmin Fenix 6X Pro | 999.99 | 2 | 10 |
| 78 | Garmin Forerunner 245 | 299.99 | 2 | 10 |
| 79 | Garmin Forerunner 945 | 499.99 | 5 | 20 |
| 80 | GlamGlow Supermud Clearing Treatment | 59 | 2 | 10 |
| 81 | Glossier Boy Brow | 16 | 3 | 10 |
| 82 | Glossier Cloud Paint | 18 | 2 | 10 |
| 83 | GoPro HERO10 Black | 399.99 | 4 | 20 |
| 84 | GoPro HERO9 Black | 449.99 | 2 | 10 |
| 85 | Gone Girl by Gillian Flynn | 22.99 | 3 | 10 |
| 86 | Google Nest Hub Max | 229.99 | 3 | 10 |
| 87 | Google Nest Wifi Router | 169 | 2 | 10 |
| 88 | Google Pixel 6 Pro | 899.99 | 2 | 10 |
| 89 | Google Pixelbook Go | 649.99 | 2 | 10 |
| 90 | H&M Slim Fit Jeans | 39.99 | 5 | 20 |
| 91 | HP Spectre x360 Laptop | 1599.99 | 2 | 10 |
| 92 | Hamilton Beach FlexBrew Coffee Maker | 89.99 | 2 | 10 |
| 93 | Hanes ComfortSoft T-Shirt | 9.99 | 15 | 50 |
| 94 | Harry Potter and the Sorcerer's Stone | 24.99 | 4 | 20 |
| 95 | Hydro Flask Standard Mouth Water Bottle | 32.95 | 5 | 20 |
| 96 | Hydro Flask Wide Mouth Water Bottle | 39.95 | 6 | 20 |
| 97 | Hyperice Hypervolt Massager | 349 | 2 | 10 |
| 98 | Instant Pot Duo | 89.99 | 4 | 20 |
| 99 | Instant Pot Duo Crisp | 179.99 | 2 | 10 |
| 100 | Instant Pot Duo Evo Plus | 139.99 | 3 | 10 |
| 101 | Instant Pot Duo Nova | 99.95 | 2 | 10 |
| 102 | Instant Pot Ultra | 139.99 | 2 | 10 |
| 103 | Keurig K-Elite Coffee Maker | 189.99 | 5 | 20 |
| 104 | Keurig K-Mini Coffee Maker | 79.99 | 3 | 10 |
| 105 | Kiehl's Midnight Recovery Concentrate | 82 | 2 | 10 |
| 106 | Kindle Paperwhite | 129.99 | 3 | 10 |
| 107 | KitchenAid Artisan Stand Mixer | 499.99 | 2 | 10 |
| 108 | KitchenAid Stand Mixer | 379.99 | 2 | 10 |
| 109 | L'Occitane Shea Butter Hand Cream | 29 | 3 | 10 |
| 110 | L'Oreal Revitalift Serum | 39.99 | 3 | 10 |
| 111 | LG OLED TV | 1299.99 | 3 | 10 |
| 112 | La Mer Cr_¨me de la Mer Moisturizer | 190 | 2 | 10 |
| 113 | Lancome La Vie Est Belle | 102 | 2 | 10 |
| 114 | Laneige Water Sleeping Mask | 25 | 2 | 10 |
| 115 | Levi's 501 Jeans | 69.99 | 5 | 20 |
| 116 | Levi's 511 Slim Fit Jeans | 59.99 | 5 | 20 |
| 117 | Levi's Sherpa Trucker Jacket | 98 | 3 | 10 |
| 118 | Levi's Trucker Jacket | 89.99 | 3 | 10 |

---

**Complete Model:**

**Sets:**  
$\mathcal{I} = \{0, 1, \ldots, 118\}$ (119 products, see table above for mapping)

**Parameters:**  
For each $i \in \mathcal{I}$,  
$A_i$ = revenue per unit (see table)  
$d_i$ = demand (see table)  
$I_i$ = initial inventory (see table)

**Variables:**  
For each $i \in \mathcal{I}$,  
$x_i \in \mathbb{Z}_+, \quad 0 \leq x_i \leq \min\{I_i, d_i\}$

**Objective:**  
$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

**Constraints:**  
For all $i \in \mathcal{I}$,
$$
x_i \leq I_i \\
x_i \leq d_i \\
x_i \geq 0 \\
x_i \in \mathbb{Z}
$$

All parameters and indices are as listed in the table above.