# 🤖 The Meta-User: How cGAN Personalizes Recommendations

## What is a GAN?

A **GAN** (*Generative Adversarial Network*) is a type of artificial intelligence composed of **two neural networks that train in competition**:

| Component | Role | Analogy |
|---|---|---|
| **Generator (G)** | Creates candidate responses | A money counterfeiter |
| **Discriminator (D)** | Evaluates if the response is good or not | A bank police officer |

The Generator tries to fool the Discriminator. The Discriminator tries not to be fooled. Over time, the Generator gets so good that it learns to generate high-quality responses — without ever explicitly seeing the right answer.

---

## Why do we use a cGAN here?

In Killswitch Engage, the cGAN (**Conditional** GAN) is used as a **Layer 4 Meta-Learning** layer. It doesn't recommend games directly. It answers a more subtle question:

> **"What is the ideal cutoff threshold to recommend games to *this* specific user?"**

The recommendation system uses a "threshold" (0.3 to 0.8) to decide if a game is relevant enough to appear in the list. A low threshold = more games, less precision. A high threshold = fewer games, more precision.

The cGAN automatically learns **what value to use for each player profile**.

---

## How does it work in practice?

```text
User Profile (KMeans Cluster + Statistics)
         │
         ▼
 ┌───────────────────┐
 │  ConditionEncoder │  ← Compresses the profile into a dense vector
 └───────────┬───────┘
             │
             ├──── + Random Noise (Z)
             │
             ▼
    ┌─────────────┐
    │  Generator  │  ← Proposes a threshold (e.g., 0.57)
    └──────┬──────┘
           │
           ▼
    ┌──────────────────┐
    │  Discriminator   │  ← "Does this threshold look optimal for this profile?"
    └──────────────────┘
           │
    ┌──────┴──────┐
    │  Real/Fake? │  ← Compares with historical validated thresholds
    └─────────────┘
```

---

## The 3 Recommendation Archetypes

The condition passed to the cGAN includes the mode chosen by the user:

| Mode | Typical Threshold | Characteristics |
|---|---|---|
| 🛡️ **Conservative** | ~0.70 | Only hits, recommends few games |
| ⚖️ **Balanced** | ~0.55 | Balance between precision and discovery |
| 🗺️ **Adventurous** | ~0.35 | Explores a lot, accepts more risks |

---

## Why not just fix the threshold?

Because different players have different tolerances. A veteran RPG player has a very high quality profile and rejects average games. A casual player wants variety and accepts more speculative recommendations.

The cGAN learns these patterns **from the historical behaviors of each cluster**, generating a personalized threshold in **~4ms** per request.

---

## Proven Results

- **PR-AUC of the full system:** `0.9153`
- **Precision@10 with cGAN:** `0.70` (~7 out of 10 right recommendations)
- **NDCG@10:** `0.8006`
- **cGAN latency overhead:** `~4ms`
- **Projected Precision@K gain:** `+20%` vs fixed threshold

---

## Internal References

- Source code: [`scripts/meta_learning/cgan_model.py`](../../scripts/meta_learning/cgan_model.py)
- Hyperparameters: [`scripts/meta_learning/hyperparameter_analysis.md`](../../scripts/meta_learning/hyperparameter_analysis.md)
- Training: [`scripts/meta_learning/train_cgan.py`](../../scripts/meta_learning/train_cgan.py)
- Training Curves: [`reports/figures/training_curves.png`](../figures/training_curves.png)

---

## 💡 Future Ideas (Next Iteration)

### cGAN as a Synthetic Meta-User Generator

The original vision for Phase 6 was to use the cGAN even more ambitiously: instead of just calibrating thresholds, it would **generate entire synthetic user profiles** from latent noise.

These generated users would be the true **"meta-users"** — archetypes learned directly from the real dataset's distribution, representing patterns that perhaps KMeans alone couldn't capture:

```text
Random Noise Z  →  [cGAN Generator]  →  Synthetic User Profile
                                        (genres, hours, frequency, cluster)
                           ↕
                  [cGAN Discriminator]  ←  Real Profiles from Dataset
```

**Role of the Generator:**
The Generator would aim to **generalize the patterns** of the real dataset to create convincing user profiles. It never sees the real profiles directly — it only receives the error signal from the Discriminator and adjusts. The goal is to produce up to a **maximum of N plausible synthetic users**, without memorizing any existing profile.

**Role of the Discriminator:**
The Discriminator **does not approve or pass on** the generated users as real accounts. Its only role is to **correct the Generator's mistakes** — detect when a synthetic profile is implausible (e.g., a user with 5000h in RPG but zero achievements) and punish the Generator for it. When training finishes, the Discriminator is discarded and only the Generator is used in production.

**Why it wasn't implemented:**
The Supabase database (free tier) reached its storage limit with the 10,000 real users and 309k sessions. Adding synthetically generated users would require a larger instance or an offline export pipeline.

**Potential impact:**
- Enrich cold start for new users with high-quality synthetic profiles
- Test the recommender against never-before-seen user patterns
- Discovery of emerging user segments that the current dataset does not cover
