# Interpretable Data Science for Smart Energy Demand Response  
### Behavioural Segmentation, Event-Centric LSTM Modelling & Decision Support

This repository contains a **portfolio-ready version** of my MSc final project at  
**Imperial College London**, developed in collaboration with the **POWBAL** initiative.

The project investigates **household demand-response behaviour** using smart-switch
events and smart-meter data, with a strong focus on **interpretability and
decision support**, rather than black-box forecasting.

> ⚠️ **Data note**  
> The original dataset used in this project is **restricted**
> (academic–industry collaboration) and **cannot be shared publicly**.  
> This repository provides the **full analytical methodology, pipeline structure,
> and modelling code**, but **does not include proprietary data**.

---

## Why this project matters

Modern demand-response programmes require more than accurate forecasts.
They require **trustworthy behavioural segmentation**, **event-level
interpretability**, and **transparent prioritisation rules** that align analytics
with real operational decisions.

This project is designed to support **where-to-act decisions**
(e.g. where to pilot interventions, how to scale programmes, how to manage risk),
rather than optimise predictive performance in isolation.

---

## Project overview

The analysis is structured around three tightly connected components:

### 1. Behavioural profiling (who)
- Construction of interpretable behavioural features
  (routine, stability, variability)
- Unsupervised segmentation into **five behavioural groups**
- Autoencoder-based latent representation used as a **risk / complexity signal**

### 2. Event-centric modelling (when & why)
- Sequence modelling around scheduled switch-off events
- **LSTM with attention** to capture short-horizon dynamics
- Class-imbalance handling, calibration, and interpretability
  (attention timelines, occlusion analysis)

### 3. Decision layer (where to act)
- Combination of opportunity, behavioural stability, and risk signals
- **Transparent ranking of segments** to answer *where to act first*
- Emphasis on **decision support and interpretability**, not individual-level scoring

---

## Data scale (indicative)

The original analysis was conducted on:
- **Over 7 million time-stamped observations**
- **30-minute temporal resolution**
- Multiple households observed over extended periods
- Integrated with contextual covariates (e.g. calendar and external signals)

Exact figures are omitted due to data-sharing restrictions.

---

## Project layout

- `src/powbal/`  
  Reusable Python package containing preprocessing pipelines,
  feature engineering, and modelling utilities.

- Jupyter notebooks (repository root)  
  Used for EDA, behavioural clustering, sequence construction,
  model training, evaluation, and interpretability analysis.

- `data/`  
  Placeholder folder (no proprietary datasets included).  
  See `data/README.md` for guidance.

---

## Running the project

The project is designed to be explored primarily through **Jupyter notebooks**,
which document the full analytical workflow end-to-end.

Typical workflow:
1. Exploratory analysis and feature construction
2. Behavioural clustering and representation learning
3. Sequence building around demand-response events
4. LSTM training, evaluation, calibration, and interpretability

The notebooks import reusable components from `src/powbal/`.

---

## Dependencies

This project uses a standard Python data-science and machine-learning stack,
including:

- `numpy`, `pandas`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `torch`
- `tqdm`
- `jupyter`

Exact versions are not pinned, as the repository is intended to demonstrate
**analytical structure and methodological reasoning**
rather than serve as a drop-in production environment.

---

## Data availability & restrictions

- Original POWBAL datasets are **not included**
- Data access was granted under restricted academic–industry agreements
- The dataset **must not be redistributed**

The methodology and code can be adapted to:
- other smart-meter or demand-response datasets
- synthetic or mock datasets
- aggregated or anonymised data (where permitted)

For academic or professional review, I am happy to provide a
**methodology walkthrough**.

---

## Outputs

Some notebooks generate intermediate artefacts locally
(e.g. processed tables, model outputs, sequence arrays).

These artefacts are:
- generated during execution
- intentionally excluded from version control via `.gitignore`

This keeps the repository lightweight and focused on methodology.

---

## Limitations

- Results depend on a restricted dataset that is not included
- Interpretability outputs (attention, occlusion) are **associative**, not causal
- Operational conclusions should be validated via controlled field experiments

---

## License

**Code:** MIT License  
**Data:** Not included (subject to original provider’s terms)

---

## Contact

**Mattia Giurisato**  
MSc Geo-Energy with Data Science & Machine Learning  
Imperial College London  

📧 Email: mattia.giurisato@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/mattia-giurisato/