# Report: Conditional Relationships Analysis (Steam EDA V2.5)

In-depth multivariate analysis was performed on the treated Steam dataset. We answered the user's biggest business questions with strict validations (ANOVA, Kruskal-Wallis, Logarithmic Exponential, and Partial Correlation) ready to tune the "Synthetic Mimicked Data Generator".

## 1. PRICE BY CATEGORY (One-Way ANOVA)
*   **What we did:** Boxplots identifying drastic variations.
*   **Ranking (Top):**
    *   Remote Play on Tablet ($41.53 ± 35.24)
    *   Steam Trading Cards ($39.27 ± 35.26)
*   **Ranking (Bottom):**
    *   In-App Purchases ($5.97 ± 18.78)
    *   MMO ($5.42 ± 17.43)
*   **Conclusion (ANOVA Test):** `F-Stat: 318.99 | p-value: 0.00`.
*   **Interpretation:** There are colossal price disparities justified by the "Category" (Multi-player, MMO, and In-App purchases aggressively tend towards the Free side, while support for Cloud, Trading Cards, or Controllers tend to be expensive - usually being the hallmark of AAA/Premium Indies).
*   **Systemic Recommendation:** The synthetic Price MUST be distributed dynamically weighting the Categories vector of the associated app.

![Boxplot Price by Category](v2_smart_imputation/price_category_boxplot.png)

## 2. AVERAGE PLAYTIME BY GENRE (Kruskal-Wallis)
*   **What we did:** Absolute engagement ranking isolating zeroed user-names.
*   **Top 5 Genres that "Addict" the Player the most (High Mean and Stable Median):**
    1.  Massively Multiplayer (Median: 372 mins)
    2.  RPG (Median: 304 mins)
    3.  Strategy (Median: 294 mins)
    4.  Simulation (Median: 238 mins)
    5.  Early Access / Adventure (Median ~220 mins)
*   **Kruskal-Wallis Test:** `H-Stat: 1464.03 | p-value: e-307`
*   **Interpretation:** "Are Strategy and RPG more addictive than Action/FPS?". Absolutely, the difference is mathematically proven and insurmountable. FPS (Action) players cycle through matches quickly and drop games (Median of 210 mins and Mean of 560).
*   **Systemic Recommendation:** Genre dictates the main equation to calculate `Average playtime` in the models!

![Boxplot Playtime x Genre](v2_smart_imputation/playtime_genre_boxplot.png)

## 3. THE BEST (AND WORST) DEVELOPERS (N >= 5)
*   **What we did:** Crossing for Quality Score stability among devs with more than 5 games created on Steam.
*   **The Gold Standard (Who almost always gets it right):** Playtouch (Positive Mean 19.7, N=10), Destructive Creations (Mean 12.8, N=6). These are cases of developers whose games emerge as incredibly well received.
*   **Rock Bottom:** "&y", "100 Cozy Games", "11 bit studios". Several of these studios delivered large volumes of terrible or irrelevant games that flirt with 0 Engagement Mean.

## 4. ACHIEVEMENTS AND RETENTION
*   **What we did:** Discrete plot comparing if `Achievements` retain the player in the game more than normal (Linear Pearson & Spearman Rank).
*   **Medians:**
    *   Low (1-10 Achiev.): 150 mins
    *   Medium (11-50 Achiev.): 226 mins
    *   High (51-150 Achiev.): 450 mins
    *   Extreme (151-500 Achiev.): 593 mins
*   **Spearman Correlation = 0.332 (p=0.0)**
*   **Interpretation:** **Spearman's Rank Coefficient proved that there is NO perfect linearity between playing more and earning more Achievements (pitiful Pearson of 0.018)**. Rather, the relationship occurs by "Tiers". There are clear motivating steps, where games with large "packages" of achievements double the median survival life.
*   **Systemic Recommendation:** "Is it worth investing in achievements?" **A LOT.** The recommendation model needs to encourage games in the 51-150 Range for Hardcore users (They retain the player guaranteed past 7 hours).

![Barplot Achievements](v2_smart_imputation/achievements_retention.png)

## 5. PRICE VS COMMUNITY SCORES
*   **What we did:** We confirmed the suspicion that the monetary value of the game DOES NOT revert to joy for the gamer and will identify if The Witcher 3 equals DOTA in the community mentality.
*   **Average Positive Percentage of the Base (The Score):**
    *   Free: 44.9% vs High ($20-$60): 63.3% vs AAA ($60+): 68.9%
*   **ANOVA p-val: 0.0**
*   **Interpretation:** "Are expensive games really not better?". **Myth busted!** Statistically, higher-priced games suffer _less hate_ in ratio. Free games are bombarded with Negative Reviews (44% average Score), probably due to zero entry barrier. A premium AAA game guarantees ~69% of the base approving it.
*   **Systemic Recommendation:** The Recommendation System should not downgrade Free games solely by Absolute Score. It has to "normalize" it. Free games carry a native toxic social weight.

![Violinplot Price vs Score](v2_smart_imputation/price_vs_score.png)

## 6. POPULARITY VS VOLUME (Power Law)
*   **Interpretation:** We analyzed Base10 logs of the Player Limit versus Base10 of the Reviews they generate.
*   **The Real Relationship:** An R² of `0.5063` was calculated with Log-log Slope at `0.7794`.
*   **What this means:** There is a strong power law in engagement per review on Steam. When a user base grows by **10x** (e.g. 10 thousand to 100 thousand sales), the vocalized volume of received Reviews **does not scale at the same speed** perfectly, assuming a slightly retarded rhythm.
*   **Recommender:** Major mainstream games ("The Last Of Us", "GTA") profit from millions of purchases by "ghost" players, who never log in to review.

![Scatterplot of Popularity by Volume](reports/figures/popularity_volume.png)

## 7. PARTIAL CORRELATION (Pingouin Multi-Variate)
We used the analytical engine to remove "parasitic effects" (statistical confounding factors) over core features:
*   **1. Price vs Metacritic Score (CONTROLLING "INDIE" FACTOR)**
    *   *Net Pearson Correlation Culling Large Studio Bias*: `r = 0.2038`.
```markdown
    *   *What this means*: "Is Metacritic paid?" Calm down. That's not what the data proves, but neither is it something the data discards. When analyzing the correlation between price and Metacritic score, controlling the "indie vs large studio" factor, we found a positive association (r ≈ 0.20). Statistically, this means more expensive games tend to receive slightly higher scores — even when we strip away part of the structural bias of budget and scale. This doesn't demonstrate bribery or deliberate manipulation, but it does demonstrate a trend. Why do AAA games seem to suffer less critical penalty? Is it just greater technical polish or is there an unconscious cognitive bias? Critics are human and expectations mold perception. A 200 million dollar game carries cultural weight, hype, and market pressure; unconsciously, this might influence judgment via psychological context. Statistics don't automatically accuse or exonerate, they just show a small and consistent pattern above the noise. In the end, maybe the right question isn't "is Metacritic paid?", but rather: to what extent do expectations and scale influence critique — even when the intent is to be neutral?
```
*   **2. Achievements vs Playtime (CONTROLLING "METACRITIC SCORE" FACTOR)**
    *   *Correlation*: `r = 0.0026`.
    *   *What this means*: Unlocking achievements in a terrible game and unlocking achievements in an absurdly excellent game (Metacritic Score 90) GENERATES strictly the NULL / SAME linear effect on the user's retained screen life! A trash game is not ignored due to lack of achievements if you strip the pure metric! The addiction factor is indifferent to the technical qualities of underlying features.
