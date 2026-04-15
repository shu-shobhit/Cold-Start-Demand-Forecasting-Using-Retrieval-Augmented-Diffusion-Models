## Chronos-Mini Baseline Summary

- Dataset: Visuelle 2.0
    
- Model: amazon/chronos-t5-mini
    
- Number of forecast samples: 50
    
- Note: All metrics are computed on the sales channel only.
    

|n_obs|RMSE|MAE|WAPE%|CRPS|CRPS_sum|
|--:|--:|--:|--:|--:|--:|
|1|1.1070|0.7734|111.0%|0.9838|0.9838|
|2|1.0983|0.7215|104.6%|0.9490|0.9490|
|3|1.0691|0.6925|102.7%|0.8551|0.8551|
|4|1.0451|0.6740|103.7%|0.8246|0.8246|

## Cold_RATD Baseline Summary

- Dataset: Visuelle 2.0
    
- Model: Cold_RATD
    
- Number of forecast samples: 5
    
- Note: All metrics are computed on the sales channel only.
    

|n_obs|RMSE|MAE|WAPE%|CRPS|CRPS_sum|
|---|--:|--:|--:|--:|--:|
|cold|1.012702|0.600488|83.03|0.652989|0.652989|
|1-shot|1.010356|0.609530|87.48|0.683238|0.683238|
|2-shot|1.012820|0.605111|87.72|0.681309|0.681309|
|3-shot|1.003438|0.597303|88.58|0.687983|0.687983|
|4-shot|0.995454|0.590141|90.82|0.708769|0.708769|

## KNN_Mean Baseline Summary

- Dataset: Visuelle 2.0
    
- Model: KNN_Mean Baseline
    
- Note: All metrics are computed on the sales channel only.
    

|Model|RMSE|MAE|WAPE%|CRPS|CRPS_sum|
|---|--:|--:|--:|--:|--:|
|global_mean|1.2394|0.8681|88.9%|0.7201|0.5611|
|knn_mean|1.3138|0.9602|92.4%|0.7966|0.5681|

