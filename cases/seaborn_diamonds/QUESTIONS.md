# Code Reading Questions for seaborn_diamonds

- [diamonds_01_eda.py](#questions-diamonds_01_edapy)
- [diamonds_02_regression.py](#questions-diamonds_02_regressionpy)
- [diamonds_03_dimreduction.py](#questions-diamonds_03_dimreductionpy)

## Questions `diamonds_01_eda.py`

**Script topics** · EDA: Distributions · EDA: Correlations · EDA: Data Quality

**Q1** · `eda.general`

- In cell [2], 20 rows have `x = y = z = 0` and are dropped, shrinking the dataset from 53,940 to 53,920 rows.
- Could zero width, length, or depth be plausible for real-world diamonds?
- Why is dropping these rows safer than leaving them in for a model that uses `x`, `y`, `z` as predictors?

**Q2** · `eda.general`

- Cell [3] prints price median $2,401 and mean $3,931, and plots both the raw and `np.log1p(price)` histograms.
- What does the mean-median gap tell you about the shape of the `price` distribution?
- Given what the histograms looks like: Which feature might be better for a later regression: `price` in raw USD or the derived feature `log1p(price)`?

**Q3** · `eda.general`

- Cell [4]'s heatmap and printed correlations show `carat` (0.92), `x` (0.89), `z` (0.87), and `y` (0.87) as the top correlators of `price`, while the correlation coefficients of `table` (0.13) and `depth` (-0.01) are close to zero.
- `carat`, `x`, `y`, and `z` are all highly correlated with each other as well as with price. What problem could this cause for an ordinary least squares regression that includes all four as separate features?
- Why do `table` and `depth` correlate so weakly with price despite being framed as "quality" measurements (see the [README](README.md) for a description of the attributes)?

**Q4** · `eda.confounding`

- Cell [5] finds that Fair-cut diamonds have a higher median price ($3282) than Ideal-cut diamonds ($1810), the reverse of what "quality" would suggest.
- What variable, not shown in this bar chart, could be driving this reversal?
- If a model regressed price on cut grade alone (ignoring carat), would the (one-hot) coefficient on "Fair" be misleading? Why?

**Q5** · `eda.general` · `process.simplicity`

- Cell [6] plots carat vs. price and reports `r(carat, price) = 0.922`, while the subplot title calls the relationship "non-linear."
- A linear correlation of 0.922 is very strong. What in the scatter plot's shape justifies still calling the relationship non-linear?
- Given how strong this single-feature relationship already is, what would the simplest possible price-prediction model look like, and how much prediction error would you expect it to leave behind (in terms of R2)?

## Questions `diamonds_02_regression.py`

**Script topics** · Linear Regression · Structural Cleaning and Encoding · Scaling and Imputation · Baselines

**Q1** · `transform.feature-engineering`

- Cell [3] one-hot encodes `cut`, `color`, `clarity` with `pd.get_dummies(..., drop_first=True)`, producing a (53920, 23) feature matrix (6 numeric + 4 cut + 6 color + 7 clarity dummies).
- Why does `drop_first=True` remove one level per categorical group instead of keeping all levels?
- What would happen to the design matrix, and to `LinearRegression`, if `drop_first=False` were used instead?

**Q2** · `pipeline.validation`

- Cell [4] uses a single 80/20 `train_test_split` (43,136 train / 10,784 test rows) with no separate validation set.
- The comment says this is because "no hyperparameter search" is performed. What changes to the analysis could make a validation split (or cross-validation) necessary?

**Q3** · `pipeline.data-leakage`

- Cell [5] fits `StandardScaler` only on `X_train[:, num_idx]`, then applies the same fitted `scaler` to both train and test through the `scale()` helper; the one-hot dummy columns are left untouched.
- What would be wrong with calling `StandardScaler().fit_transform()` on the full dataset before the train/test split?
- Why are the dummy columns excluded from `num_idx` and left as 0/1 instead of being standardized too?

**Q4** · `evaluation.baseline` · `process.controlled-change`

- Cell [6]'s carat-only baseline scores RMSE $1518 / R² 0.856; cell [7]'s full 23-feature model scores RMSE $1136 / R² 0.920. This is a 25.2% RMSE reduction.
- The R² of the carat-only baseline (0.856) is close to `r(carat, price)² ≈ 0.922² ≈ 0.850` from the EDA script. Why should that relationship hold for simple linear regression?
- Going from the baseline to the full model adds 19 new features (cut/color/clarity dummies plus depth/table/x/y/z) all at once. Why does that make it harder to say which specific feature(s) drove the improvement?

**Q5** · `deployment.prod-fit`

- Cell [8] reports that the full model predicts negative prices for 7.0% of training rows and 6.6% of test rows.
- Why are predictions of negative prices a problem? What is the underlying structural weakness of the model? Explain your reasoning.
- What assumption of ordinary linear regression are violated by the distribution of the real-world price data?

**Q6** · `eda.general`

- Cell [9]'s predicted-vs-actual plot shows points "hugging" the dashed diagonal at low prices but fanning out well above and below it as actual price rises past roughly $10,000.
- What statistical assumption of OLS does this fanning pattern violate?
- Which kind of diamond (in terms of carat) is most likely to fall in the widely-scattered region of this plot, based on what you know from the EDA script?

**Q7** · `explain.coeff-meaning` · `explain.importance`

- Cell [10]'s coefficient plot ranks `carat` (+$5,415) as by far the largest positive standardized coefficient, with `clarity_I1` (-$5,461) as the largest-magnitude coefficient of any sign.
- Since all numeric features were z-scored in cell [5], can you directly compare the magnitude of the `carat` coefficient to the magnitude of a categorical dummy coefficient like `clarity_I1`?
- The coefficient for `cut_Good` is relative to which omitted category? Does a positive or negative sign there mean "better than Fair" or "better than no cut at all"?
- Given that `carat`, `x`, `y`, `z` are all highly correlated (cell [4] of the EDA script), how much should you trust the exact size of the `carat` coefficient versus the general fact that "size dominates"?

## Questions `diamonds_03_dimreduction.py`

**Script topics** · Unsupervised Learning · Structural Cleaning and Encoding · EDA: Correlations · Comparing PCA and t-SNE for dimension reduction and data understanding

**Q1** · `transform.feature-engineering`

- Cell [3] ordinally encodes `cut`, `color`, `clarity` as integer ranks (`cut_ord`, `color_ord`, `clarity_ord`) instead of one-hot encoding them as in the regression script, giving a (53920, 9) feature matrix instead of 23 columns.
- Why is ordinal encoding a reasonable choice here whereas it was avoided (in favor of one-hot) in `diamonds_02_regression.py`?
- What would go wrong if `CUT_ORDER`, `COLOR_ORDER`, or `CLARITY_ORDER` listed a grade out of its true quality order?

**Q2** · `modeling.algorithm`

- Cell [4]'s scree plot shows PC1 (the first principal component) alone explains 47.2% of variance, and 5 of 9 components are needed to reach the 90% threshold line.
- What quantity is PCA actually maximizing when it chooses the direction of PC1, and where in `sklearn`'s `PCA` is that computed?
- Given that `carat`, `x`, `y`, `z` were shown to be highly collinear in the EDA script, why is it unsurprising that a single component captures nearly half the total variance here?

**Q3** · `explain.importance`

- Cell [5] colors the PC1-vs-PC2 scatter by `log1p(price)` (price itself was never used to fit the PCA) and reports `r(PC1, log-price) = 0.913`.
- In cell [3], `price` / `log_price` were excluded from `ALL_FEATURES` that are used for the PCA. What does it mean that PC1 still correlates so strongly with it?
- Does this result tell you more about the diamonds dataset, or about the fact that carat/size so strongly determines price in general?

**Q4** · `modeling.algorithm`

- Cell [6] uses t-SNE as an alternative to visualize high-dimensional space in 2D (compared to the two principal components from cell [5]). What are the advantages and drawbacks of PCA vs. t-SNE?

**Q5** · `process.reproducibility`

- Cell [6] draws a fixed-size subsample (`TSNE_SAMPLE = 5_000`) using `rng = np.random.default_rng(RANDOM_STATE)` before fitting `TSNE(..., random_state=RANDOM_STATE)`.
- Which parts of the t-SNE output would change if `RANDOM_STATE` were removed from both the `rng.choice(...)` call and the `TSNE(...)` call?
- Why does this script subsample to 5,000 rows for t-SNE in cell [6], while PCA in cell [4] runs on the full 53,920 rows?

**Q6** · `extension.alternatives`

- Cell [7] reports `r(t-SNE 1, log-price) = 0.843`, lower than PCA's `r(PC1, log-price) = 0.913` from cell [5].
- Given that the non-linear t-SNE projection aligns with price no better (in fact slightly worse) than the linear PCA projection, what does that suggest about the type of structure separating diamonds by price?
- When would you use any of these visualizations? Would any of those be suitable as a quick visual proxy for price if you had to hand this over to a colleague - or would you choose another visualization? Explain why.

**Q7** · `eda.data-sampling`

- Cell [8]'s printed counts show the 5,000-row t-SNE subsample contains 2,003 Ideal-cut diamonds but only 161 Fair-cut diamonds, and the resulting scatter shows Ideal-cut points forming a distinct cluster while Fair/Good/Very-Good/Premium points seem to mix more together.
- Could the sparser, more scattered appearance of the Fair-cut points be partly an artefact of having far fewer of them in the subsample, rather than evidence that Fair diamonds truly lack structure?
- How would you check whether the clustering pattern seen here is stable, versus an accident of which 5,000 rows were drawn?
