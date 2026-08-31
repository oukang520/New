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
