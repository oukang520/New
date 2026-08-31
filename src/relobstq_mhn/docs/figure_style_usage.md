# Shared Figure Style Usage

This project keeps top-journal figure design defaults in one place:

- `configs/figure_style.yaml`: typography, palettes, output formats, benchmark references and figure rules.
- `src/figure_style.py`: Python helpers for loading the style, applying matplotlib defaults and saving PDF/PNG outputs.
- `configs/experiment_registry.yaml`: experiment-to-config registry for Experiments 1-7 and future experiment defaults.

The shared style also stores user-provided reference-figure design patterns.
These references are used only for layout, annotation and information-density
logic. Their colors are not imported; the project palette in
`configs/figure_style.yaml` remains the only active palette.

## Minimal Pattern For New Experiments

Add this top-level field to the experiment config:

```yaml
plot_style_config: configs/figure_style.yaml
```

In the experiment runner:

```python
import figure_style

figure_style.configure_matplotlib(config)
figure_style.save_figure(fig, output_base, config)
```

Use the shared source list and rules in design review reports:

```python
sources = figure_style.design_sources_markdown(config)
rules = figure_style.design_rules_markdown(config)
patterns = figure_style.design_patterns_markdown(config)
external_skills = figure_style.external_skill_sources_markdown(config)
```

## Design Policy

Future main figures should default to the shared style and should only deviate
when the scientific question requires a different visual grammar. Deviations
should be documented in the experiment's figure design review.

Preferred reusable patterns include compact in-panel scale keys, direct
annotation, boxed context insets when they clarify a metric, shared-axis small
multiples, raw-point overlays for replicate summaries, phase bands/arrows for
ordered processes, uncertainty bands for fitted trends and compact panel
letters for composites.

## GitHub Skill Review

The current shared policy incorporates design and engineering guardrails from:

- K-Dense `scientific-visualization`: scientific integrity, uncertainty,
  missing-data handling, accessibility, export discipline and rendered-output
  inspection.
- K-Dense `matplotlib`: object-oriented Matplotlib layouts, GridSpec control
  and publication export practices.
- K-Dense `seaborn`: statistical visualization semantics, long-form data,
  uncertainty display and categorical comparison rules.
- OpenAI curated `screenshot`, `pdf` and `jupyter-notebook`: installed as
  auxiliary inspection/iteration skills for later Codex turns.

These external sources are used as design-process references only. The active
project palette remains the user-selected five-color palette stored in
`configs/figure_style.yaml`.

## Rendered Figure Audit

After regenerating final figures, run:

```bash
python src/audit_experiment_figures.py
```

For the manuscript-level figure set only, run:

```bash
python src/audit_experiment_figures.py --main-only --contact-sheet --output-dir reports/main_figure_style_audit
```

The script writes:

- `reports/figure_style_audit/figure_style_audit.csv`
- `reports/figure_style_audit/figure_style_audit.md`
- `reports/main_figure_style_audit/main_figure_contact_sheet.png` when
  `--contact-sheet` is used

The audit checks PNG size, near-blank images, extreme aspect ratios and possible
border clipping. A `WARN` result means the figure deserves visual inspection; it
does not automatically invalidate the experiment.
