# 🗺️ Llama-3 Financial Analyst — 20-Phase Project Roadmap

## Legend
- ✅ Done | 🔄 In Progress | ⏳ Today | 🔜 Next | ❌ Not Started

---

## PHASE 1 — Project Foundation ✅
**Goal:** Dependencies, configs, virtual environment
- [x] `pyproject.toml` — all dependencies defined
- [x] `configs/training_config.yaml` — QLoRA training config
- [x] `uv venv` + `.venv` activated
- [ ] `configs/inference_config.yaml`
- [ ] `configs/monitoring_config.yaml`
- [ ] `.env.example` — environment variables template

---

## PHASE 2 — Data Ingestion ✅
**Goal:** Download + merge financial datasets from HuggingFace
- [x] `src/data_pipeline/ingestion.py`
  - Downloads `gbharti/finance-alpaca` (68K samples)
  - Downloads `virattt/financial-qa-10K`
  - Normalizes to unified schema `{instruction, input, output}`
  - Merges + shuffles + saves to `data/raw/`

---

## PHASE 3 — Data Validation ✅
**Goal:** Quality gates on raw data
- [x] `src/data_pipeline/validator.py`
  - Length checks (min/max)
  - Duplicate detection
  - ValidationReport with stats
  - Filters bad samples

---

## PHASE 4 — PII Masking ⏳ TODAY
**Goal:** Remove sensitive financial data (account numbers, names)
- [ ] `src/data_pipeline/pii_masker.py`
  - Presidio-based PII detection
  - Mask: PERSON, PHONE, EMAIL, CREDIT_CARD, US_SSN
  - Returns clean dataset

---

## PHASE 5 — Data Formatting ⏳ TODAY
**Goal:** Convert to Llama-3 Instruct chat template
- [ ] `src/data_pipeline/formatter.py`
  - Apply `<|begin_of_text|>...<|end_of_text|>` format
  - System prompt: "You are an expert financial analyst..."
  - Handle instruction + optional input + output
  - Save formatted dataset to `data/processed/`

---

## PHASE 6 — Tokenization ⏳ TODAY
**Goal:** Tokenize formatted text for training
- [ ] `src/data_pipeline/tokenizer.py`
  - Load tokenizer from HuggingFace
  - Tokenize with max_length=2048
  - Add padding + truncation
  - Save tokenized dataset
- [ ] `src/data_pipeline/__init__.py` — expose pipeline

---

## PHASE 7 — Training Config Loader ⏳ TODAY
**Goal:** Load + validate YAML configs for training
- [ ] `src/training/config_loader.py`
  - Load `training_config.yaml`
  - Build `BitsAndBytesConfig` for 4-bit quantization
  - Build `LoraConfig` for QLoRA adapters
  - Build `TrainingArguments`
- [ ] `src/training/__init__.py`

---

## PHASE 8 — QLoRA Trainer ⏳ TODAY
**Goal:** Core training loop — the heart of the project
- [ ] `src/training/qlora_trainer.py`
  - Load base model (Llama-3.2-3B) in 4-bit
  - Apply LoRA adapters via PEFT
  - Use `SFTTrainer` from TRL
  - Train + eval loop
  - Save adapter weights to `models/`
  - Push to HuggingFace Hub

---

## PHASE 9 — Training Callbacks
**Goal:** Logging, checkpointing, early stopping
- [ ] `src/training/callbacks.py`
  - WandB logging callback
  - MLflow tracking
  - Custom loss logger
  - Early stopping on eval_loss plateau
- [ ] `src/training/sweep.py` — Optuna hyperparameter tuning

---

## PHASE 10 — Model Evaluation
**Goal:** Measure model quality post-training
- [ ] `src/evaluation/benchmarks.py`
  - ROUGE-L score on held-out financial QA
  - BERTScore for semantic similarity
  - Custom financial accuracy metric
- [ ] `src/evaluation/report_generator.py`
  - Generate HTML evaluation report
  - Save to `reports/`

---

## PHASE 11 — Regression Gate
**Goal:** Block bad models from deployment
- [ ] `src/evaluation/regression_gate.py`
  - Compare new model vs baseline
  - If ROUGE drops >5% → fail gate
  - Used in CI/CD pipeline

---

## PHASE 12 — Inference Engine
**Goal:** Load trained model + generate responses
- [ ] `src/inference/engine.py`
  - Load base model + LoRA adapters
  - 4-bit quantized inference
  - Streaming token generation
  - Batch inference support

---

## PHASE 13 — FastAPI Backend
**Goal:** Production REST API for the model
- [ ] `src/inference/api.py`
  - `POST /predict` — single inference
  - `POST /predict/stream` — SSE streaming
  - `GET /health` — health check
  - `GET /metrics` — Prometheus metrics endpoint
  - Request/Response Pydantic models

---

## PHASE 14 — API Security
**Goal:** Auth + rate limiting + caching
- [ ] `src/inference/auth.py` — JWT API key validation
- [ ] `src/inference/rate_limiter.py` — SlowAPI 100 req/min
- [ ] `src/inference/cache.py` — Redis response cache (TTL=1hr)

---

## PHASE 15 — Monitoring Stack
**Goal:** Observe model in production
- [ ] `src/monitoring/metrics.py` — Prometheus counters/histograms
- [ ] `src/monitoring/drift_detector.py` — Output quality drift
- [ ] `src/monitoring/alerting.py` — Slack/email alerts

---

## PHASE 16 — DVC Pipeline
**Goal:** Reproducible ML pipeline with data versioning
- [ ] `dvc.yaml` — stages: ingest → validate → format → tokenize → train → evaluate
- [ ] `dvc.lock` — generated after first run
- [ ] `data/.gitignore` — DVC tracked data

---

## PHASE 17 — Docker
**Goal:** Containerize training + inference
- [ ] `docker/Dockerfile.training` — training container
- [ ] `docker/Dockerfile.inference` — FastAPI inference container
- [ ] `docker/docker-compose.yaml` — inference + Redis + Prometheus + Grafana

---

## PHASE 18 — CI/CD GitHub Actions
**Goal:** Automated testing + deployment
- [ ] `.github/workflows/ci.yaml` — lint + tests on every PR
- [ ] `.github/workflows/training.yaml` — trigger Kaggle training
- [ ] `.github/workflows/deploy.yaml` — build Docker + push to registry

---

## PHASE 19 — Kubernetes Deployment
**Goal:** Production-grade scaling
- [ ] `k8s/deployment.yaml` — inference deployment
- [ ] `k8s/hpa.yaml` — auto-scaling on CPU/memory
- [ ] `k8s/ingress.yaml` — external traffic routing
- [ ] `k8s/helm/` — Helm chart for one-command deploy

---

## PHASE 20 — Portfolio Polish
**Goal:** GitHub-ready, interview-ready
- [ ] `README.md` — full professional README with badges, demo, architecture
- [ ] `ARCHITECTURE.md` — update with actual diagrams
- [ ] `notebooks/demo.ipynb` — interactive demo
- [ ] Demo video recorded
- [ ] Deployed on cloud (AWS/GCP/Render)
- [ ] HuggingFace model card published

---

## 📊 Today's Target (6 Hours)

| Phase | Task | Est. Time | Status |
|-------|------|-----------|--------|
| 1 | Foundation (configs complete) | 20 min | ✅ |
| 2 | Data Ingestion | Done | ✅ |
| 3 | Data Validation | Done | ✅ |
| 4 | PII Masking | 30 min | ⏳ |
| 5 | Data Formatting | 45 min | ⏳ |
| 6 | Tokenization | 30 min | ⏳ |
| 7 | Config Loader | 40 min | ⏳ |
| 8 | QLoRA Trainer | 90 min | ⏳ |

**By 3 AM → Phases 1-8 done = Data Pipeline + Training Core = TRUE 50%**
