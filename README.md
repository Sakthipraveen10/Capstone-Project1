# Instamart Customer Intelligence & Support Copilot

## Overview

One repository that turns messy synthetic order exports into a trusted SQLite database (Part A), segments customers and predicts lapse (Part B), triages support tickets with privacy and cost guardrails (Part C) and answers policy questions and decides refunds with a retrieval assistant and a tool-calling agent (Part D). All data is synthetic and fictional, all money is in INR, and every language-model call goes through the offline `mock_llm.py` (no key, no network, no cost). `generate_data.py` and `mock_llm.py` are the supplied files, committed unmodified. Allowed libraries only: standard library, pandas, numpy, scikit-learn, joblib.

| Path | Purpose |
|---|---|
| `src/a1_audit.py` ... `src/a5_explore.py` | Part A: audit, cleaning + storage, validation, SQL runner, pandas exploration |
| `sql/queries.sql` | the seven Part A queries |
| `src/b1_segments.py`, `src/b2_lapse.py` | Part B: RFM, K-means, PCA, lapse model |
| `src/c1_triage.py` | Part C: triage pipeline |
| `src/rag.py`, `src/agent.py` | Part D: policy RAG, tools, ReAct agent, resume |
| `outputs/`, `state/`, `data/clean/`, `data/instamart.db` | committed results |

## Reproduce

Run from the repository root (Python 3.11+).

1. `python -m venv .venv`
2. `source .venv/bin/activate`   (Windows: `.venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. `python generate_data.py`   (last line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`)
5. `python src/a1_audit.py`
6. `python src/a2_clean_store.py`
7. `python src/a3_validate.py`   (prints `ALL CHECKPOINTS PASSED`)
8. `python src/a4_sql.py`
9. `python src/a5_explore.py`
10. `python src/b1_segments.py`
11. `python src/b2_lapse.py`
12. `python src/c1_triage.py | tee outputs/triage_run.log`
13. `python src/rag.py`
14. `python src/agent.py`

Seeds: `random_state=42` everywhere in scikit-learn, `np.random.RandomState(42)` for over-sampling, `random.seed(42)` for retry jitter. Note on the pinned versions: the build sandbox had no network access, so `requirements.txt` lists the four direct dependencies pinned to the versions the results were produced with, rather than a literal `pip freeze` of a fresh environment.

## Part A

### Audit (task 2)

Shapes: customers (1200, 5), products (40, 4), orders raw (13084, 9). Missing values (count, %): customers.age 61 (5.08%), customers.gender 30 (2.50%); orders.delivery_minutes 2,541 (19.42%); every other column 0. `order_amount` is read as text because pandas would silently turn the planted string `N/A` into NaN; its 120 unparseable entries are counted after `pd.to_numeric(errors="coerce")`. Dtypes: ids/text are strings, `age` float64 (blanks), `delivery_minutes` float64, `quantity`, `discount_pct`, `price_inr` int64.

| Audit figure | Observed | Reference |
|---|---|---|
| raw order rows | 13,084 | 13,084 |
| exact duplicate rows | 119 | 119 |
| pending rows | 656 | 656 |
| distinct order_id | 12,309 | 12,309 |
| non-numeric order_amount | 120 | 120 |
| negative delivery_minutes | 106 | 106 |
| blank delivery_minutes | 2,541 | 2,541 |
| blank age | 61 | 61 |
| ages outside 10-100 | 6 | 6 |
| distinct raw city spellings | 24 | 24 |
| distinct non-blank gender spellings | 10 | 10 |
| blank gender | 30 | 30 |

### Cleaning rules and what each changed (tasks 3-4)

Customers: (a) city stripped and title-cased: 24 spellings become 6 (Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184). (b) gender mapped male/m to `male`, female/f to `female`, blank to `unknown`: female 610, male 560, unknown 30. (c) 6 impossible ages set to missing, then all 67 missing ages (61 blank + 6) filled with the city median via `groupby().transform("median")`; city medians ending in .5 are rounded to whole years, giving 0 missing and ages 18-60. (d) `signup_date` converted to datetime.

Orders, in this order: (1) `order_ts` to datetime. (2) 119 exact duplicate rows removed (13,084 to 12,965). (3) 656 earlier `pending` records removed by keeping the record with the latest `order_ts` per `order_id` (to 12,309 rows). (4) `order_amount` to numeric: 112 unparseable values remain after de-duplication and are refilled as `round(price_inr x quantity x (1 - discount_pct/100), 2)`. (5) 105 negative delivery times set to missing; for completed orders 774 missing delivery times (negatives plus originally blank) are filled with the median of the product category among completed orders; the 1,199 cancelled orders keep a missing delivery time because a cancelled order was never delivered.

Result: 12,309 rows (11,110 completed, 1,199 cancelled), no pending rows, 0 missing amounts, 0 missing delivery among completed, 1,199 among cancelled, completed revenue INR 2,998,668.80. All match the references. `src/a3_validate.py` asserts 15 checkpoints and prints `ALL CHECKPOINTS PASSED`. Clean tables are in `data/clean/*.csv` and in `data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` stored as `YYYY-MM-DD HH:MM:SS` text).

### SQL results (task 6; `sql/queries.sql`)

| Query | Result | Reference |
|---|---|---|
| Q1 WHERE + HAVING | 30 customers | 30 |
| Q2 three-table join | 30 rows, total INR 2,998,668.80, largest Bengaluru / Electronics INR 233,909.45 | same |
| Q3 scalar subquery | 11 products above average price | 11 |
| Q4 LEFT JOIN | 25 customers without a completed order | 25 |
| Q5 CTEs | 550 customers above the average of INR 2,552.06 | 550 / 2,552.06 |
| Q6 window ranks | top per city: Bengaluru C0082 12,856.20; Chennai C0930 10,151.40; Delhi C0291 13,742.60; Hyderabad C0743 12,730.05; Mumbai C0693 14,766.55; Pune C0426 10,377.10; 8 rows with RANK() != ROW_NUMBER() | same |
| Q7 running total | 183 daily rows, final cumulative INR 2,998,668.80 | same |

Why ties make RANK() differ from ROW_NUMBER(): customers with exactly the same spend in a city share one RANK() value (and the next rank is skipped), while ROW_NUMBER() always gives every row a unique number, so after the first tie the two columns disagree.

### pandas cross-check and exploration (task 7)

`pd.pivot_table(..., margins=True)` grand total = INR 2,998,668.80, identical to SQL Q2. Per city (sum / mean / count of completed `order_amount`): Bengaluru 523,643.40 / 276.47 / 1,894; Chennai 463,629.15 / 272.72 / 1,700; Delhi 502,829.80 / 243.62 / 2,064; Hyderabad 528,589.90 / 284.95 / 1,855; Mumbai 479,058.50 / 297.00 / 1,613; Pune 500,918.05 / 252.48 / 1,984.

| Statistic (completed order_amount) | Value |
|---|---|
| mean / median | 269.91 / 135.00 |
| skewness | 4.61 |
| Q1 / Q3 | 64.80 / 297.00 |
| fences (1.5 x IQR) | lower -283.50, upper 645.30 |
| outliers below / above | 0 / 897 |

Correlations over completed orders: order_amount-discount_pct -0.095, order_amount-quantity +0.087, order_amount-delivery_minutes -0.003, discount_pct-quantity 0.000, discount_pct-delivery_minutes -0.010, quantity-delivery_minutes -0.013.

*What is the data about?* Six months (April-September 2025) of grocery, dairy, snack, personal-care and electronics orders from 1,175 customers across six cities. *Distributions:* order values are strongly right-skewed (mean 269.91 far above median 135.00, skew 4.61) because a few electronics baskets are large. *Outliers:* 897 orders exceed the upper fence of 645.30, none fall below; these are genuine electronics purchases, not errors. *Relationships:* all correlations are tiny; the largest in absolute value is order_amount with discount_pct (-0.095), just ahead of order_amount with quantity (+0.087). A correlation of this size is very weak and does not establish causation.

## Part B

### RFM (task 1, snapshot 2025-10-01 00:00:00)

1,175 customers; recency median 19, min 0, max 180; frequency mean 9.46, max 33; monetary mean 2,552.06, max 14,766.55. All equal the references.

### K-means (task 2; StandardScaler, `n_init=10`, `random_state=42`)

| K | inertia | silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Chosen K = 4.** The elbow is at K=4: inertia falls by 525 from K=3 to K=4 but only by 126 from K=4 to K=5, and K=4 has the highest silhouette (0.526). Four segments is also a number a retention team can run four distinct campaigns for. Inertia is the sum of squared distances to the nearest centre, so adding clusters can only keep it equal or lower, which is why K=8 always beats K=2; in the limit of one cluster per customer it reaches zero, so a low inertia alone only rewards over-splitting and cannot pick K without the elbow, the silhouette and interpretability.

### Segment profile, names and PCA (task 3)

Cluster numbers are arbitrary labels with no ordering; names are assigned by rule from the profile.

| id | name | size | mean recency (days) | mean frequency | mean monetary (INR) | action |
|---|---|---|---|---|---|---|
| 0 | Light Regulars | 436 | 28.9 | 3.8 | 1,057.6 | bundle and basket-builder offers |
| 1 | **Premium Big Spenders** (`is_premium`) | 138 | 24.1 | 6.8 | 7,129.7 | priority support, early access to electronics |
| 2 | Loyal Frequent Buyers | 419 | 8.8 | 19.4 | 3,410.0 | loyalty rewards, protect the habit |
| 3 | Fading Occasional Buyers | 182 | 109.0 | 2.2 | 686.0 | win-back coupon |

PCA on the scaled RFM matrix: explained variance ratio 0.634 and 0.218, total 0.852. `outputs/customer_segments.csv` has the required columns; `is_premium` is true only for the highest-mean-monetary segment (138 customers).

### Lapse problem and preprocessing (tasks 4-5)

Cutoff 2025-09-01; cohort = customers with a completed order before the cutoff: **1,156 customers, 443 lapsed (38.32%)**; split 924 train / 232 test (`stratify=y`, `random_state=42`). Features (orders before the cutoff only): recency, frequency, monetary, average discount, cancellation rate, average delivery minutes, favourite category, city, age, gender, tenure days. No order dated 2025-09-01 or later entered a feature, and the task-3 segment labels were not used as features. Baseline "always not lapsed": accuracy 61.6%, lapsed-class F1 0.

Preprocessing is a scikit-learn `Pipeline` (`StandardScaler` for numbers, `OneHotEncoder(drop="first")` for city, gender, favourite category). No encoder, scaler or resampler saw test rows: they are fitted inside the pipeline on training data only (inside each fold during grid search). The learned scaler mean of frequency is 8.115 against a full-cohort mean of 8.029; they differ because the scaler was fitted on the 924 training rows only, while the cohort mean also includes the 232 test rows. A code assertion confirms no feature column has a missing value.

### Tuning and test evaluation (task 6)

| Model | best parameters | best CV F1 |
|---|---|---|
| LogisticRegression (`C` in 0.01..100) | C = 100 | 0.651 |
| RandomForest (`max_depth` 3/5/8 x `min_samples_leaf` 5/10/20) | depth 8, leaf 10 | 0.687 |

Logistic regression's best C is 100 here whereas the reference run had C=1; that is expected because my feature set differs from the reference (extra features). Test set, lapsed = positive class:

| Model | TN | FP | FN | TP | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| baseline (never lapsed) | 143 | 0 | 89 | 0 | 0 | 0 | 0 |
| LogisticRegression | 111 | 32 | 31 | 58 | 0.644 | 0.652 | 0.648 |
| RandomForest | 111 | 32 | 32 | 57 | 0.640 | 0.640 | 0.640 |

**Selected model: RandomForest.** It was selected on cross-validated F1 (0.687 versus 0.651, training data only), not on the test set; the two models are practically tied on the test set (0.640 versus 0.648), and the forest is more robust to the strongly correlated frequency and monetary features. Missing a lapsing customer means losing all of their future revenue silently, whereas wrongly targeting a loyal customer costs only a coupon or a message, so recall matters at least as much as precision; the chosen model has a test F1 of 0.640, above the 0.55 requirement.

### Class-balance experiment (task 7)

Training set over-sampled with replacement (lapsed rows only) to 570 / 570; best logistic regression refitted; evaluated on the unchanged test set:

| Model | precision | recall | F1 |
|---|---|---|---|
| LogisticRegression | 0.644 | 0.652 | 0.648 |
| LogisticRegression + over-sampling | 0.589 | 0.854 | 0.697 |

Recall moved most (+0.20, 58 to 76 true positives, FN 31 to 13) while precision fell slightly (-0.055, FP 32 to 53). For the retention team this means catching far more lapsing customers at the price of contacting more customers who would have stayed, which is the right trade when the retention message is cheap.

### Interpretation, persistence, retention list (task 8)

Top-5 random-forest importances: monetary 0.278, frequency 0.277, recency 0.111, average delivery minutes 0.062, average discount 0.053. Business reading: how much and how often a customer buys matter most; the more recently and the better served (fast delivery) they are, the less likely they lapse. Logistic regression coefficients (standardised features): frequency -1.70 (more orders, much less likely to lapse), monetary +0.28 (given the same frequency, higher spend is slightly associated with lapsing; frequency and monetary are correlated, so read this with care), cancellation rate -0.12 (small and not decisive). Recency is +0.04, essentially zero once frequency is in the model.

The selected model is saved to `outputs/lapse_model.joblib`; the reloaded model reproduces the original test predictions exactly (checked in code). `outputs/retention_list.csv` holds the top 10 of the 40 Premium customers in the test set by predicted lapse probability; **1 of the 10** truly lapsed. The Premium flag comes from whole-window RFM that includes the September window defining the label, so this list is illustrative and not a leakage-free evaluation.

## Part C

### Guardrails, redaction, prompts (tasks 1-3)

4 empty tickets (`skipped_empty`) and 4 long tickets (734 estimated tokens each, above `MAX_INPUT_TOKENS` = 400, `flagged_long`) never reach the model. Redaction substitutions over all 164 tickets: 55 emails, 91 phone numbers, 132 order ids, **278 in total**; re-applying the three patterns to the redacted texts finds 0. A code assertion checks that no prompt sent to the model matches any pattern. The user message is exactly the redacted text. The zero-shot prompt (26 estimated tokens) and the few-shot prompt (114 estimated tokens, with an `EXAMPLES:` line and four labelled examples including a fraud one) both contain `CLASSIFY`, list the four categories and demand JSON only.

### Retries, validation (tasks 4-5)

Each full run over the 156 eligible tickets produces exactly 19 `TransientLLMError`s, all recovered at retry number 0 with waits between 0.0453 and 0.0562 s (limit 0.045-0.0575 s); one log line per retry, for example `retry log: ticket=T0002 retry=0 wait=0.0530s` (full log in `outputs/triage_run.log`). No ticket ends with a permanent error. Status counts for every full run: **135 classified_json, 12 classified_keyword, 9 flagged_invalid_output** (plus 4 skipped_empty, 4 flagged_long).

### Zero-shot versus few-shot (task 6)

| Run | classified | accuracy | total tokens | tokens per classified ticket |
|---|---|---|---|---|
| zero-shot | 147 | 85.7% (126/147) | 8,464 | 57.6 |
| few-shot | 147 | 100% (147/147) | 22,192 | 151.0 |

Zero-shot confusion (rows true, columns predicted): billing 38/0/0/0; delivery 0/36/0/0; fraud 11 predicted billing and 23 correct; technical 10 predicted billing and 29 correct. The few-shot prompt costs about 93 extra tokens per ticket (+13,728 tokens in total, 2.6x) but removes all 21 errors, roughly 650 tokens per corrected ticket. It is worth paying whenever a wrong label is expensive, as here: 11 fraud tickets sent to billing would miss the 2-hour security SLA. For a low-stakes, high-volume classification the cheaper zero-shot prompt would be enough.

### Budget, routing and output (task 7)

Budget demo (zero-shot, `TOKEN_BUDGET = 6000`, `outputs/triage_budget_demo.csv`): 6,027 tokens used, **45 tickets ended `budget_exhausted`** (97 classified_json, 8 classified_keyword, 6 flagged_invalid_output, 4 empty, 4 long). Final run (`outputs/triage_results.csv`, few-shot, `TOKEN_BUDGET = 100000`): 164 rows; routes: security_team/HIGH 34, premium_desk/HIGH 11, support_queue/NORMAL 102, human_review/NORMAL 17. Rule-validated by the code (JSON or keyword accepted): 147 tickets. Sent to humans: 17 (9 invalid outputs, 4 empty, 4 long); no ticket was guessed. Customers with no row in the segment file are treated as not premium.

## Part D

### Chunking, retrieval, grounding (tasks 1-4)

15 chunks of 40 words with overlap 10: refund 4, delivery 3, escalation 3, payment 3, privacy 2. For `refund policy late delivery` the top three chunks are all from `refund_policy.txt` (scores 0.272, 0.251, 0.245). TF measures how often a term occurs in a chunk and IDF down-weights terms that appear in many chunks, so words that are rare and specific to one chunk dominate the similarity. A limitation is that TF-IDF only matches words: two phrasings with the same meaning but different words (for example "money back" versus "refund") get a low similarity.

| Evaluation | overlap 10 | overlap 0 (13 chunks) |
|---|---|---|
| answerable: expected doc in top 3 | 12/12 | 12/12 |
| answerable: expected fact in answer | 12/12 | 10/12 |
| unanswerable: fallback "I could not find" | 2/2 | 2/2 |

With zero overlap two questions break: "What delivery time makes a completed order eligible for a full refund?" (the answer sentence is cut at a chunk boundary, so the context holds "receives a full refund of the order amount" without "more than 45 minutes") and "How soon is money returned when a payment fails but is deducted?" (the "48 hours" part falls in the next chunk). The 10-word overlap repeats the boundary words so each sentence stays whole in at least one chunk.

### Agent results (tasks 5-7)

| scenario | status | refund INR | model calls | tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,067 |
| S2 | REFUND | 135.00 | 5 | 1,067 |
| S3 | ESCALATE | 1999.00 | 5 | 1,054 |
| S4 | DENIED | 0.00 | 4 | 741 |
| S5 | DENIED | 0.00 | 3 | 439 |
| S6 | NOT_FOUND | 0.00 | 2 | 140 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,204 |

All refund amounts equal `expected_refund_inr` (asserted in the script). S7 finishes all six model calls under the default `token_limit=6000`.

**Stop-safely demonstrations:** S1 with `token_limit=200` ends **FALLBACK_TOKEN_LIMIT** (456 tokens after 3 model calls, the third requested tool was not executed); S1 with `time_limit_seconds=0` ends **FALLBACK_TIMEOUT** (0 model calls, before the first call). Both return a human-handoff summary in `final_answer`.

**Crash and resume (S1):** crash raised right after the second tool's checkpoint (`state/S1-crash-demo.json`), then `resume_agent("S1-crash-demo")`. Tools executed across crash and resume, in order: `get_order`, `search_policy` (before the crash), `calculate_refund`, `issue_refund` (after resume): four executions, none repeated; final status REFUND, INR 128.25, 5 model calls in total. Note: `outputs/refund_ledger.jsonl` has 4 lines (S1, S2 issued, S3 escalated, plus S1 again from the crash demo, because the demo is a second run of the same request); a production system would make `issue_refund` idempotent per order id. `outputs/agent_runs.jsonl` has the 7 scenario runs; `outputs/agent_demos.jsonl` has the three demonstrations.

### ReAct versus plan-and-execute (task 8)

My ReAct agent asks the model once per step: 5 calls for a successful refund (S1), 4 for an on-time order (S4), 3 for a cancelled order (S5) and 2 for a missing order (S6), i.e. 140 to 1,067 tokens because the growing message history is re-sent every call. A plan-and-execute design asks the model once for the whole list (get_order, search_policy, calculate_refund, issue/escalate) plus one final answer, so about 2 calls for S1 and, since the history is not re-sent after every step, fewer tokens (my estimate, not measured). The cost is rigidity: when an observation is unexpected, such as NOT_FOUND (S6) or a cancelled order (S5), the pre-written plan keeps running a calculate-and-issue step that makes no sense, so it needs a replanning call or a validity check after each step; ReAct simply stops after 2 or 3 calls because it reacts to what it saw. For this refund workflow I would choose ReAct, or a hybrid with a fixed plan and an early-exit check after `get_order`: the workflow is short (at most 5 calls), refunds move money so every step should be checked against the real observation, the per-call overhead is small, and the safety limits (iterations, tokens, time) bound the cost of a loop like S7.

## Limitations and Next Steps

- All data is synthetic; segments, lapse behaviour and ticket categories come from simple generator rules, so real-world accuracy would be lower.
- The Premium flag is a post-hoc whole-window label (its RFM includes the September window that defines the lapse label), so `retention_list.csv` is illustrative and not a leakage-free evaluation. The model's own features use only orders before 2025-09-01.
- Test set is small (232 rows, 89 lapsed): F1 differences of 0.01 between models are noise. Over-sampling by duplication can overfit duplicated rows; a probability threshold or class weights would be a cleaner alternative.
- The mock LLM classifies by keywords and the agent follows a fixed script, so the accuracies and call counts say more about the guardrail code than about a real model. TF-IDF retrieval is lexical; a real system would add embeddings.
- The refund ledger is not idempotent, and the 7-day window is assumed satisfied in every simulated request.
- The sandbox used to build this had no network, so `requirements.txt` was written from the installed versions instead of from a fresh `pip freeze`.
- Next steps: temporal cross-validation for the lapse model, a leakage-free premium definition computed only from pre-cutoff data, idempotent refunds, and an automated test suite for the pipelines.
