Let $I$ be the set of products, indexed by $i$, with the following data for each product $i$:

- Product Name: as listed below
- Revenue per unit: $r_i$
- Demand: $d_i$
- Initial Inventory: $s_i$

Decision variables:

$x_i \in [0, \min\{d_i, s_i\}]$ (continuous): number of units of product $i$ to fulfill for customer purchases.

Objective:

$\max \sum_{i \in I} r_i x_i$

Subject to:

$0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I$

where $r_i$, $d_i$, $s_i$ are as follows (source order):

| $i$ | Product Name | $r_i$ (Revenue) | $d_i$ (Demand) | $s_i$ (Initial Inventory) |
|---|-------------------------------|---------|--------|---------------------|
| 1 | 1984 by George Orwell | 19.99 | 6 | 20 |
| 2 | Adidas 3-Stripes Shorts | 29.99 | 7 | 30 |
| 3 | Adidas Essential Track Pants | 44.99 | 5 | 20 |
| 4 | Adidas FIFA World Cup Football | 29.99 | 4 | 20 |
| 5 | Adidas Originals Superstar Sneakers | 79.99 | 6 | 20 |
| 6 | Adidas Originals Trefoil Hoodie | 64.99 | 6 | 20 |
| 7 | Adidas Ultraboost Running Shoes | 179.99 | 3 | 10 |
| 8 | Adidas Ultraboost Shoes | 179.99 | 3 | 10 |
| 9 | Amazon Echo Dot (4th Gen) | 49.99 | 5 | 20 |
| 10 | Amazon Echo Show 10 | 249.99 | 2 | 10 |
| 11 | Amazon Fire TV Stick 4K | 49.99 | 5 | 20 |
| 12 | Anastasia Beverly Hills Brow Wiz | 23 | 3 | 10 |
| 13 | Anker PowerCore Portable Charger | 59.99 | 6 | 20 |
| 14 | Anova Precision Cooker | 199 | 3 | 10 |
| 15 | Anova Precision Oven | 599 | 2 | 10 |
| 16 | Apple AirPods Max | 549 | 2 | 10 |
| 17 | Apple AirPods Pro | 249.99 | 3 | 10 |
| 18 | Apple MacBook Air | 1199.99 | 2 | 10 |
| 19 | Apple MacBook Pro 16-inch | 2399 | 2 | 10 |
| 20 | Apple TV 4K | 179 | 3 | 10 |
| 21 | Apple Watch Series 8 | 399.99 | 4 | 20 |
| 22 | Apple iPad Air | 599.99 | 3 | 10 |
| 23 | Atomic Habits by James Clear | 16.99 | 6 | 20 |
| 24 | Babolat Pure Drive Tennis Racket | 199.99 | 5 | 20 |
| 25 | Becoming by Michelle Obama | 32.5 | 6 | 20 |
| 26 | Biore UV Aqua Rich Watery Essence Sunscreen | 15 | 2 | 10 |
| 27 | Blueair Classic 480i | 599.99 | 3 | 10 |
| 28 | Bose QuietComfort 35 Headphones | 299.99 | 2 | 10 |
| 29 | Bose QuietComfort 35 II Wireless Headphones | 299 | 2 | 10 |
| 30 | Bose SoundLink Color Bluetooth Speaker II | 129 | 2 | 10 |
| 31 | Bose SoundLink Revolve+ Speaker | 299.99 | 5 | 20 |
| 32 | Bose SoundSport Wireless Earbuds | 149.99 | 3 | 10 |
| 33 | Bowflex SelectTech 1090 Adjustable Dumbbells | 699.99 | 2 | 10 |
| 34 | Bowflex SelectTech 552 Dumbbells | 399.99 | 2 | 10 |
| 35 | Breville Nespresso Creatista Plus | 499.95 | 2 | 10 |
| 36 | Breville Smart Coffee Grinder Pro | 199.95 | 2 | 10 |
| 37 | Breville Smart Grill | 299.95 | 3 | 10 |
| 38 | Breville Smart Oven | 299.99 | 2 | 10 |
| 39 | Calvin Klein Boxer Briefs | 29.99 | 7 | 30 |
| 40 | Canon EOS R5 Camera | 3899.99 | 2 | 10 |
| 41 | Canon EOS Rebel T7i DSLR Camera | 749.99 | 2 | 10 |
| 42 | Caudalie Vinoperfect Radiance Serum | 79 | 2 | 10 |
| 43 | CeraVe Hydrating Facial Cleanser | 14.99 | 3 | 10 |
| 44 | Champion Reverse Weave Hoodie | 49.99 | 5 | 20 |
| 45 | Chanel No. 5 Perfume | 129.99 | 2 | 10 |
| 46 | Charlotte Tilbury Magic Cream | 100 | 2 | 10 |
| 47 | Clinique Dramatically Different Moisturizing Lotion | 29.5 | 2 | 10 |
| 48 | Clinique Moisture Surge | 52 | 2 | 10 |
| 49 | Columbia Fleece Jacket | 59.99 | 6 | 20 |
| 50 | Crock-Pot 6-Quart Slow Cooker | 49.99 | 3 | 10 |
| 51 | Cuisinart Coffee Center | 199.95 | 3 | 10 |
| 52 | Cuisinart Custom 14-Cup Food Processor | 199.99 | 2 | 10 |
| 53 | Cuisinart Griddler Deluxe | 159.99 | 2 | 10 |
| 54 | De'Longhi Magnifica Espresso Machine | 899.99 | 2 | 10 |
| 55 | Dr. Jart+ Cicapair Tiger Grass Color Correcting Treatment | 52 | 2 | 10 |
| 56 | Drunk Elephant C-Firma Day Serum | 78 | 2 | 10 |
| 57 | Dune by Frank Herbert | 25.99 | 6 | 20 |
| 58 | Dyson Pure Cool Link | 499.99 | 2 | 10 |
| 59 | Dyson Supersonic Hair Dryer | 399.99 | 5 | 20 |
| 60 | Dyson V11 Vacuum | 499.99 | 2 | 10 |
| 61 | Dyson V8 Absolute | 399.99 | 2 | 10 |
| 62 | Educated by Tara Westover | 28 | 4 | 20 |
| 63 | Estee Lauder Advanced Night Repair | 105 | 2 | 10 |
| 64 | Eufy RoboVac 11S | 219.99 | 5 | 20 |
| 65 | Fenty Beauty Killawatt Highlighter | 36 | 2 | 10 |
| 66 | First Aid Beauty Ultra Repair Cream | 34 | 3 | 10 |
| 67 | Fitbit Charge 5 | 129.99 | 3 | 10 |
| 68 | Fitbit Inspire 2 | 99.95 | 3 | 10 |
| 69 | Fitbit Luxe | 149.95 | 3 | 10 |
| 70 | Fitbit Versa 3 | 229.95 | 5 | 20 |
| 71 | Forever 21 Graphic Tee | 12.99 | 7 | 30 |
| 72 | Fresh Sugar Lip Treatment | 24 | 2 | 10 |
| 73 | Gap 1969 Original Fit Jeans | 59.99 | 5 | 20 |
| 74 | Gap Crewneck Sweatshirt | 34.99 | 6 | 20 |
| 75 | Gap Essential Crewneck T-Shirt | 19.99 | 8 | 30 |
| 76 | Gap High Rise Skinny Jeans | 49.99 | 5 | 20 |
| 77 | Garmin Edge 530 | 299.99 | 3 | 10 |
| 78 | Garmin Fenix 6X Pro | 999.99 | 2 | 10 |
| 79 | Garmin Forerunner 245 | 299.99 | 2 | 10 |
| 80 | Garmin Forerunner 945 | 499.99 | 5 | 20 |
| 81 | GlamGlow Supermud Clearing Treatment | 59 | 2 | 10 |
| 82 | Glossier Boy Brow | 16 | 3 | 10 |
| 83 | Glossier Cloud Paint | 18 | 2 | 10 |
| 84 | GoPro HERO10 Black | 399.99 | 4 | 20 |
| 85 | GoPro HERO9 Black | 449.99 | 2 | 10 |
| 86 | Gone Girl by Gillian Flynn | 22.99 | 3 | 10 |
| 87 | Google Nest Hub Max | 229.99 | 3 | 10 |
| 88 | Google Nest Wifi Router | 169 | 2 | 10 |
| 89 | Google Pixel 6 Pro | 899.99 | 2 | 10 |
| 90 | Google Pixelbook Go | 649.99 | 2 | 10 |
| 91 | H&M Slim Fit Jeans | 39.99 | 5 | 20 |
| 92 | HP Spectre x360 Laptop | 1599.99 | 2 | 10 |
| 93 | Hamilton Beach FlexBrew Coffee Maker | 89.99 | 2 | 10 |
| 94 | Hanes ComfortSoft T-Shirt | 9.99 | 15 | 50 |
| 95 | Harry Potter and the Sorcerer's Stone | 24.99 | 4 | 20 |
| 96 | Hydro Flask Standard Mouth Water Bottle | 32.95 | 5 | 20 |
| 97 | Hydro Flask Wide Mouth Water Bottle | 39.95 | 6 | 20 |
| 98 | Hyperice Hypervolt Massager | 349 | 2 | 10 |
| 99 | Instant Pot Duo | 89.99 | 4 | 20 |
| 100 | Instant Pot Duo Crisp | 179.99 | 2 | 10 |
| 101 | Instant Pot Duo Evo Plus | 139.99 | 3 | 10 |
| 102 | Instant Pot Duo Nova | 99.95 | 2 | 10 |
| 103 | Instant Pot Ultra | 139.99 | 2 | 10 |
| 104 | Keurig K-Elite Coffee Maker | 189.99 | 5 | 20 |
| 105 | Keurig K-Mini Coffee Maker | 79.99 | 3 | 10 |
| 106 | Kiehl's Midnight Recovery Concentrate | 82 | 2 | 10 |
| 107 | Kindle Paperwhite | 129.99 | 3 | 10 |
| 108 | KitchenAid Artisan Stand Mixer | 499.99 | 2 | 10 |
| 109 | KitchenAid Stand Mixer | 379.99 | 2 | 10 |
| 110 | L'Occitane Shea Butter Hand Cream | 29 | 3 | 10 |
| 111 | L'Oreal Revitalift Serum | 39.99 | 3 | 10 |
| 112 | LG OLED TV | 1299.99 | 3 | 10 |
| 113 | La Mer Cr_¨me de la Mer Moisturizer | 190 | 2 | 10 |
| 114 | Lancome La Vie Est Belle | 102 | 2 | 10 |
| 115 | Laneige Water Sleeping Mask | 25 | 2 | 10 |
| 116 | Levi's 501 Jeans | 69.99 | 5 | 20 |
| 117 | Levi's 511 Slim Fit Jeans | 59.99 | 5 | 20 |
| 118 | Levi's Sherpa Trucker Jacket | 98 | 3 | 10 |
| 119 | Levi's Trucker Jacket | 89.99 | 3 | 10 |

Summary:

$\max \sum_{i=1}^{119} r_i x_i$

subject to

$0 \leq x_i \leq \min\{d_i, s_i\}$ for $i=1,\ldots,119$

with all $r_i$, $d_i$, $s_i$ as listed above in source order.