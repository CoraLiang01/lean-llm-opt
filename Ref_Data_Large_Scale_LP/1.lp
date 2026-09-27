\ Model Original_RAG_AP
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xafadc5e61dd6a8aa
Minimize
  3000 x[MA,P1] + 3200 x[MA,P2] + 3100 x[MA,P3] + 2800 x[MB,P1]
   + 3300 x[MB,P2] + 2900 x[MB,P3] + 2900 x[MC,P1] + 3100 x[MC,P2]
   + 3000 x[MC,P3]
Subject To
 worker[MA]: x[MA,P1] + x[MA,P2] + x[MA,P3] = 1
 worker[MB]: x[MB,P1] + x[MB,P2] + x[MB,P3] = 1
 worker[MC]: x[MC,P1] + x[MC,P2] + x[MC,P3] = 1
 project[P1]: x[MA,P1] + x[MB,P1] + x[MC,P1] = 1
 project[P2]: x[MA,P2] + x[MB,P2] + x[MC,P2] = 1
 project[P3]: x[MA,P3] + x[MB,P3] + x[MC,P3] = 1
Bounds
Binaries
 x[MA,P1] x[MA,P2] x[MA,P3] x[MB,P1] x[MB,P2] x[MB,P3] x[MC,P1] x[MC,P2]
 x[MC,P3]
End
