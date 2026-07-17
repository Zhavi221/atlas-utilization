# smooth + functions

339 smoothed MC20 analysis2.0 curves were used, obtained by enforcing the following conditions:
- $Z_{max}^{LR} \leq 5$
    - The first 10% of bins were excluded when evaluating $Z_{max}^{LR}$
- `num_bins` $\geq 25$
- `total_events` $\geq 100$
- `min_num_events` $\geq 0.35$

341 analytical functions were used with a similar dynamic range to the MC20 smoothed curves. 

# generation 

- *training*: 1.5M (MC20 + functions)
- *validation*: 150k (MC20 + functions)
- *application*: 300k (MC20)

*training* and *validation* datasets have signal fraction of **0.5**. 

# training

- *batch_size*: 2048
- *num_epochs*: 300

# prediction + plotting 

Predictions are done on datasets generated using only smoothed MC20 curves. For plotting details, see the `predict_plot_config.yaml` file. 