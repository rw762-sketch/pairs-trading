# 50-stock research sample

Five randomly selected stocks in each of ten randomly selected sub-industries. Seed: 42. Industry groups must have at least five training-eligible stocks. This is a peer-group sample, not a proportionally representative sample of the index. No correlations or later-period metrics were used to choose it.

Uses cached prices from the full-universe screen. Training: September 19, 2023 through October 22, 2025. Later check: October 23, 2025 through September 17, 2026.

| Industry | Five selected stocks |
| --- | --- |
| Asset Management & Custody Banks | ARES, BEN, BLK, NTRS, TROW |
| Biotechnology | ABBV, AMGN, BIIB, INCY, MRNA |
| Consumer Staples Merchandise Retail | COST, DG, DLTR, TGT, WMT |
| Health Care Services | CI, CVS, DGX, DVA, LH |
| Life Sciences Tools & Services | A, CRL, DHR, IQV, WAT |
| Multi-Family Residential REITs | CPT, ESS, MAA, UDR, VMRK |
| Packaged Foods & Meats | GIS, HSY, KHC, MKC, SJM |
| Property & Casualty Insurance | ALL, CB, HIG, PGR, TRV |
| Retail REITs | FRT, KIM, O, REG, SPG |
| Specialty Chemicals | ALB, IFF, LYB, PPG, SHW |

Rerun from the project folder:

```bash
./venv/bin/python screen_pairs.py --universe output/pair-research-50/universe.json --output output/pair-research-50
```

Each rerun saves a new dated folder under `output/pair-research-50/`.
