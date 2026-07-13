# index/config — index parameters as versioned YAML

Approved design decision: all stability-index parameters (weights, windows,
thresholds) live here as YAML files under version control, so parameter changes
are reviewable diffs rather than code edits.

Expected parameters include (mentor-agreed, 2026-07-12): per-feature min/max
distress thresholds for the Call Report features, the GP-score bands
(80 / 90 cut points for negative / neutral / positive), and the weights for
combining the fundamentals score with the sentiment score.

- Phase: 3
- Owner: TBD
