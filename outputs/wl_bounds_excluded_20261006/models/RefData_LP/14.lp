\ Canonical native-bound formulation; bounds excluded from WL graph.
Maximize
  obj: + 1818 x[Sony_Alpha_Refrigerator,1] + 1818 x[Sony_Alpha_Refrigerator,2] + 1818 x[Sony_Alpha_Refrigerator,3] + 1818 x[Sony_Alpha_Refrigerator,4] + 1818 x[Sony_Alpha_Refrigerator,5] + 1818 x[Sony_Alpha_Refrigerator,6] + 1609 x[Sony_Bravia_XR,1] + 1609 x[Sony_Bravia_XR,2] + 1609 x[Sony_Bravia_XR,3] + 1609 x[Sony_Bravia_XR,4] + 1609 x[Sony_Bravia_XR,5] + 1609 x[Sony_Bravia_XR,6] + 509 x[Sony_HT-A7000_Soundbar,1] + 509 x[Sony_HT-A7000_Soundbar,2] + 509 x[Sony_HT-A7000_Soundbar,3] + 509 x[Sony_HT-A7000_Soundbar,4] + 509 x[Sony_HT-A7000_Soundbar,5] + 509 x[Sony_HT-A7000_Soundbar,6] + 1808 x[Sony_PlayStation_5,1] + 1808 x[Sony_PlayStation_5,2] + 1808 x[Sony_PlayStation_5,3] + 1808 x[Sony_PlayStation_5,4] + 1808 x[Sony_PlayStation_5,5] + 1808 x[Sony_PlayStation_5,6] + 528 x[Sony_PlayStation_Soundbar,1] + 528 x[Sony_PlayStation_Soundbar,2]
    + 528 x[Sony_PlayStation_Soundbar,3] + 528 x[Sony_PlayStation_Soundbar,4] + 528 x[Sony_PlayStation_Soundbar,5] + 528 x[Sony_PlayStation_Soundbar,6]
Subject To
  sony_space[1]: + 400 x[Sony_Alpha_Refrigerator,1] + 200 x[Sony_Bravia_XR,1] + 40 x[Sony_HT-A7000_Soundbar,1] + 60 x[Sony_PlayStation_5,1] + 50 x[Sony_PlayStation_Soundbar,1] <= 1200
  sony_space[2]: + 400 x[Sony_Alpha_Refrigerator,2] + 200 x[Sony_Bravia_XR,2] + 40 x[Sony_HT-A7000_Soundbar,2] + 60 x[Sony_PlayStation_5,2] + 50 x[Sony_PlayStation_Soundbar,2] <= 1374
  sony_space[3]: + 400 x[Sony_Alpha_Refrigerator,3] + 200 x[Sony_Bravia_XR,3] + 40 x[Sony_HT-A7000_Soundbar,3] + 60 x[Sony_PlayStation_5,3] + 50 x[Sony_PlayStation_Soundbar,3] <= 800
  sony_space[4]: + 400 x[Sony_Alpha_Refrigerator,4] + 200 x[Sony_Bravia_XR,4] + 40 x[Sony_HT-A7000_Soundbar,4] + 60 x[Sony_PlayStation_5,4] + 50 x[Sony_PlayStation_Soundbar,4] <= 2042
  sony_space[5]: + 400 x[Sony_Alpha_Refrigerator,5] + 200 x[Sony_Bravia_XR,5] + 40 x[Sony_HT-A7000_Soundbar,5] + 60 x[Sony_PlayStation_5,5] + 50 x[Sony_PlayStation_Soundbar,5] <= 1000
  sony_space[6]: + 400 x[Sony_Alpha_Refrigerator,6] + 200 x[Sony_Bravia_XR,6] + 40 x[Sony_HT-A7000_Soundbar,6] + 60 x[Sony_PlayStation_5,6] + 50 x[Sony_PlayStation_Soundbar,6] <= 1800
Bounds
  0 <= x[Sony_Alpha_Refrigerator,1] <= +inf
  0 <= x[Sony_Alpha_Refrigerator,2] <= +inf
  0 <= x[Sony_Alpha_Refrigerator,3] <= +inf
  0 <= x[Sony_Alpha_Refrigerator,4] <= +inf
  0 <= x[Sony_Alpha_Refrigerator,5] <= +inf
  0 <= x[Sony_Alpha_Refrigerator,6] <= +inf
  0 <= x[Sony_Bravia_XR,1] <= +inf
  0 <= x[Sony_Bravia_XR,2] <= +inf
  0 <= x[Sony_Bravia_XR,3] <= +inf
  0 <= x[Sony_Bravia_XR,4] <= +inf
  0 <= x[Sony_Bravia_XR,5] <= +inf
  0 <= x[Sony_Bravia_XR,6] <= +inf
  0 <= x[Sony_HT-A7000_Soundbar,1] <= +inf
  0 <= x[Sony_HT-A7000_Soundbar,2] <= +inf
  0 <= x[Sony_HT-A7000_Soundbar,3] <= +inf
  0 <= x[Sony_HT-A7000_Soundbar,4] <= +inf
  0 <= x[Sony_HT-A7000_Soundbar,5] <= +inf
  0 <= x[Sony_HT-A7000_Soundbar,6] <= +inf
  0 <= x[Sony_PlayStation_5,1] <= +inf
  0 <= x[Sony_PlayStation_5,2] <= +inf
  0 <= x[Sony_PlayStation_5,3] <= +inf
  0 <= x[Sony_PlayStation_5,4] <= +inf
  0 <= x[Sony_PlayStation_5,5] <= +inf
  0 <= x[Sony_PlayStation_5,6] <= +inf
  0 <= x[Sony_PlayStation_Soundbar,1] <= +inf
  0 <= x[Sony_PlayStation_Soundbar,2] <= +inf
  0 <= x[Sony_PlayStation_Soundbar,3] <= +inf
  0 <= x[Sony_PlayStation_Soundbar,4] <= +inf
  0 <= x[Sony_PlayStation_Soundbar,5] <= +inf
  0 <= x[Sony_PlayStation_Soundbar,6] <= +inf
Generals
  x[Sony_Alpha_Refrigerator,1] x[Sony_Alpha_Refrigerator,2] x[Sony_Alpha_Refrigerator,3] x[Sony_Alpha_Refrigerator,4] x[Sony_Alpha_Refrigerator,5] x[Sony_Alpha_Refrigerator,6] x[Sony_Bravia_XR,1] x[Sony_Bravia_XR,2] x[Sony_Bravia_XR,3] x[Sony_Bravia_XR,4] x[Sony_Bravia_XR,5] x[Sony_Bravia_XR,6] x[Sony_HT-A7000_Soundbar,1] x[Sony_HT-A7000_Soundbar,2] x[Sony_HT-A7000_Soundbar,3] x[Sony_HT-A7000_Soundbar,4] x[Sony_HT-A7000_Soundbar,5] x[Sony_HT-A7000_Soundbar,6] x[Sony_PlayStation_5,1] x[Sony_PlayStation_5,2] x[Sony_PlayStation_5,3] x[Sony_PlayStation_5,4] x[Sony_PlayStation_5,5] x[Sony_PlayStation_5,6] x[Sony_PlayStation_Soundbar,1] x[Sony_PlayStation_Soundbar,2] x[Sony_PlayStation_Soundbar,3] x[Sony_PlayStation_Soundbar,4] x[Sony_PlayStation_Soundbar,5] x[Sony_PlayStation_Soundbar,6]
End
