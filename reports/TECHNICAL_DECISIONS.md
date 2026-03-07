# Technical Decisions — Killswitch Engage

## Layer 4: Meta-Observer (cGAN)

### Motivation
Traditional ranking models (Layer 3) usually return scores with varying scales for different users. A fixed threshold (e.g., 0.5) can be too restrictive for a "Casual" user and too permissive for a "Hardcore" user.

### Implementation (Production)
- **Architecture**: Conditional GAN (cGAN) where the Generator receives the user profile (147 features) + latent noise (32 dim) to predict the ideal threshold.
- **Integration**: cGAN was set as the default mode (`meta`) of the system.
- **Inference**: Performed in real-time via PyTorch (CPU) at the time of recommendation.
- **Validation**: The model was validated on 1,000 profiles, demonstrating automatic adjustment for highly active profiles (selective thresholds) vs sporadic ones.

## Recommendation Flow
1. **RF Filter**: Based on tags and price.
2. **KMeans Archetype**: Clustering by behavior.
3. **LightFM Ranker**: Hybrid factorization matrix.
4. **cGAN Threshold**: Meta-heuristic calibration of the score cutoff.

## Security
- All secrets strictly contained in `.env`.
- Models loaded from controlled paths in `src/config/paths.py`.
- Log sanitization to prevent PII leakage.
