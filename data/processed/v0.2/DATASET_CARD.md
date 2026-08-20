# Phase 06 SFT dataset v0.2

- Parent: `sft-v0.1`
- Total / train / validation: `795 / 715 / 80`
- Replayed unique v0.1 / removed duplicate inputs / targeted new: `479 / 21 / 316`
- Categories: `{'工具调用': 303, '无关': 69, '最终 JSON': 107, '确认': 129, '追问': 187}`
- Exact eval input overlap: `0`
- DPO data: not generated in Round 001

## Targeted additions

- Relative dates, weekdays, cross-month and cross-year normalization.
- Multi-turn minimal replacement for time, technician, preference and duration.
- Missing-information boundaries that must ask instead of calling tools.
- Available-plan acceptance and rejection with consistent booking semantics.

Date answers are computed during dataset construction only to create and validate labels. The trained model remains responsible for date understanding and normalization at inference time.
