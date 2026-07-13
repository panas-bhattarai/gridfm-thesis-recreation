# GridFM Thesis Recreation

Independent, reduced-scale recreation of the ETH Zürich Master's thesis:

> **"Foundation Model for the Power Grid"** — Matteo Mazzonelli, ETH Zürich, 2025
> Supervisors: Prof. Dr. M. El-Assady, Anna Varbella; IBM Research: Dr. Thomas Brunschwiler, Dr. Jonas Weiss
> DOI: [10.3929/ethz-b-000733077](https://doi.org/10.3929/ethz-b-000733077)

Companion position paper: *"Foundation Models for the Electric Power Grid"* (Hamann et al., [arXiv:2407.09434](https://arxiv.org/abs/2407.09434)).

> ⚠️ **This is an unofficial educational reproduction at reduced scale. It is not affiliated
> with or endorsed by ETH Zürich, IBM, or the authors, and the numbers here are NOT the
> thesis's results.** Please read **[DISCLAIMER.md](DISCLAIMER.md)** before using or citing anything.

## What this project does

The thesis pre-trains graph-based neural networks ("GridFM") on hundreds of thousands of
solved AC power-flow scenarios using self-supervised masked feature reconstruction with a
physics-informed loss (AC power balance equations). We recreate the full pipeline —
data generation (pandapower), graph datasets, models, pre-training, fine-tuning,
zero-shot evaluation — at a scale that trains on a consumer laptop.

## Citing the original work

This repository reproduces — it does not replace — the original research. If you use these
ideas, cite the thesis and paper, not this repository:

```bibtex
@mastersthesis{mazzonelli2025gridfm,
  author = {Mazzonelli, Matteo},
  title  = {Foundation Model for the Power Grid},
  school = {ETH Z\"urich},
  year   = {2025},
  doi    = {10.3929/ethz-b-000733077}
}

@article{hamann2024gridfm,
  author  = {Hamann, Hendrik F. and others},
  title   = {Foundation Models for the Electric Power Grid},
  journal = {arXiv preprint arXiv:2407.09434},
  year    = {2024}
}
```

## License

Code and notebooks in this repository are released under the [MIT License](LICENSE).
See [DISCLAIMER.md](DISCLAIMER.md) for scope, affiliation, and results caveats.
