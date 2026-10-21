# Network context for predicting cattle movements: LightGBM vs linear trend

This experiment follows up on "The application of social network mining to cattle movement analysis: introducing the predictive trend mining framework" (2016), which applies social network mining to cattle movements between premises and proposes predictive trend mining. Plain per-link trend tracking is a common way to forecast these movements, but it ignores what is happening around each link in the network. This script asks whether adding network context helps.

**Question:** Can a LightGBM model that uses network context (neighbouring-link flows and node activity) beat a standard per-link linear trend extrapolation at predicting next-period cattle movement counts between premises?

**Dataset:** A seeded synthetic premises-to-premises movement network generated with numpy inside `analysis.py`. It is not the paper's own data: the Great Britain movement records are not publicly downloadable. Results therefore do not transfer directly to real movements.

**Baseline vs improved:** The baseline extrapolates each link's recent counts with a linear trend. The improved model is a LightGBM regressor on log-transformed counts that adds network features (in/out degree and neighbouring-link flows from the previous month) to lagged counts and rolling means. Both models share the same simulated data, the same time-ordered train/test split, and the same evaluation. The main metric is mean absolute error on raw next-month counts. Top-decile hit rate and F1 on active links are secondary metrics.

**How to run**

```
pip install -r requirements.txt
python analysis.py
```

**Results:** results are written to `results.json` (and `figures/mae_comparison.png`).

## Reference

This project is motivated by the following paper:

> P Nohuddin, F Coenen, R Christley (2016). *The application of social network mining to cattle movement analysis: introducing the predictive trend mining framework*. Social Network Analysis and Mining. https://link.springer.com/article/10.1007/s13278-016-0353-x

Paper author: Prof. Frans Coenen, University of Liverpool (frans@liverpool.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
