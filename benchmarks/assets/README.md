# Benchmark assets

Fixed media for the benchmarks so runs are reproducible across machines without
depending on an external download or a transitive package's data files.

- `china.jpg`, `flower.jpg` — the two sample photographs shipped with
  scikit-learn (`sklearn/datasets/images/`), public-domain images long used as
  canonical vision test inputs. Stable input for `alloy bench --dataset
  multimodal`.
- `harvard.wav`, `jfk_1963_0626_berliner_64kb.mp3` — short public-domain speech
  clips kept as audio test inputs.
