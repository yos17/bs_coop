# Derivation And Simulation Check

This note gives a GitHub-friendly derivation of the locality theorem for the localized distributed beamforming idea, and then checks whether the current simulation results support the intuition.

Short answer: yes, with one important nuance.

- The main intuition is correct on the **true objective**: as the cooperation radius `rho` grows, the achieved weighted sum-rate approaches the globally coordinated benchmark.
- The theorem is cleanest when stated for the **true objective attained by a localized optimizer**.
- If the localized objective simply drops far-field interference terms, then the surrogate objective is optimistic. In that case, the quantity that converges from below is `F(W_rho^*)`, not `max_W F^(rho)(W)` itself.
- Stronger spatial decay helps. In the current experiments, larger path-loss exponent `alpha` makes the true gap shrink faster as `rho` increases.

## 1. Successor Formulation

We consider a cooperative multi-cell downlink network with:

- base stations `b = 1, ..., B`
- user sets `U_b`
- beamformers `w_(b,u)`
- channels `h_(k,b)`

The transmitted signal from BS `b` is

$$
x_b = \sum_{u \in U_b} w_{b,u} s_u,
\qquad \mathbb{E}|s_u|^2 = 1.
$$

The received signal at user `k` is

$$
y_k = \sum_{b=1}^B h_{k,b}^H x_b + n_k.
$$

If user `k` is served by BS `b(k)`, then

$$
\mathrm{SINR}_k(W) =
\frac{|h_{k,b(k)}^H w_{b(k),k}|^2}
{
\sum_{u \in U_{b(k)}, u \neq k} |h_{k,b(k)}^H w_{b(k),u}|^2
+ \sum_{c \neq b(k)} \sum_{u \in U_c} |h_{k,c}^H w_{c,u}|^2
+ \sigma_k^2
}.
$$

The global weighted sum-rate objective is

$$
F(W) = \sum_k \alpha_k \log_2(1 + \mathrm{SINR}_k(W)),
$$

subject to per-BS power constraints

$$
\sum_{u \in U_b} \|w_{b,u}\|^2 \le P_b.
$$

This is the successor to the 2008 Jungnickel-style decomposition:

- 2008 predecessor: fixed BD precoders + distributed power allocation
- successor here: localized adaptive beamforming + distributed power control

## 2. Localization Model

For each base station `b`, define a cooperation neighborhood

$$
C_b(\rho) = \{ c : d(b,c) \le \rho \}.
$$

Only stations in `C_b(rho)` exchange optimization variables.

For user `k`, define the omitted far-field interference

$$
\Delta I_k^{(\rho)}(W)
=
\sum_{c \notin C_{b(k)}(\rho)}
\sum_{u \in U_c}
|h_{k,c}^H w_{c,u}|^2.
$$

If we simply drop this term, we get an optimistic surrogate objective

$$
\widetilde{F}^{(\rho)}(W),
$$

which is easier to optimize locally but is not itself a lower bound on the true global utility.

## 3. Local WMMSE-Style Update

Using the standard WMMSE reformulation:

- each user updates its MMSE receiver `g_k`
- each user updates its MSE weight `v_k`
- each BS updates its beamformers using only local information
- dual variables price power usage
- interference prices penalize local leakage

The local beamformer update has the form

$$
w_{b,u}^\star = A_b^{-1}(\alpha_u v_u g_u h_{u,b}),
$$

with

$$
A_b =
\sum_k \alpha_k v_k |g_k|^2 h_{k,b} h_{k,b}^H
+ \sum_q \lambda_{b,q} E_q
+ \sum_{k \in N_b(\rho)} \pi_k h_{k,b} h_{k,b}^H.
$$

This is the clean matrix-valued successor to the old dual power-loading logic.

## 4. Correct Theorem Statement

Let

$$
F^\star = \max_W F(W).
$$

Let `W_rho^*` be any optimizer of the optimistic localized surrogate:

$$
W_\rho^\star \in \arg\max_W \widetilde{F}^{(\rho)}(W).
$$

Define the **true** objective achieved by that localized optimizer:

$$
\widehat{F}_\rho^\star := F(W_\rho^\star).
$$

Assume:

1. Path-loss decay:

$$
\|h_{k,b}\|^2 \le C_h (1 + d(k,b))^{-\alpha},
\qquad \alpha > 2.
$$

2. Bounded feasible beamformers:

$$
\sum_{u \in U_b} \|w_{b,u}\|^2 \le P_b \le P_{\max}.
$$

3. Bounded shell growth in 2D:
the number of BSs at distance in `[m, m+1)` is at most proportional to `1 + m`.

4. Positive noise floor:

$$
\sigma_k^2 \ge \sigma_{\min}^2 > 0.
$$

Then there exists a constant `K > 0` such that

$$
0 \le F^\star - \widehat{F}_\rho^\star \le K \rho^{2-\alpha}.
$$

Hence

$$
\widehat{F}_\rho^\star \to F^\star
\qquad \text{as } \rho \to \infty.
$$

If the distributed algorithm returns `W_rho^dagger` with local surrogate gap

$$
\widetilde{F}^{(\rho)}(W_\rho^\star)
- \widetilde{F}^{(\rho)}(W_\rho^\dagger)
\le \delta_\rho,
$$

then

$$
0 \le F^\star - F(W_\rho^\dagger)
\le K \rho^{2-\alpha} + \delta_\rho.
$$

So if `delta_rho -> 0`, then the algorithmic solution also approaches the global optimum.

## 5. Why The Bound Works

The proof is straightforward once the optimistic surrogate is handled correctly.

### 5.1 Far-field interference tail

By Cauchy-Schwarz and the power bound,

$$
\Delta I_k^{(\rho)}(W)
\le
\sum_{c \notin C_{b(k)}(\rho)}
\|h_{k,c}\|^2
\sum_{u \in U_c} \|w_{c,u}\|^2
\le
C_h P_{\max}
\sum_{c \notin C_{b(k)}(\rho)}
(1 + d(k,c))^{-\alpha}.
$$

In 2D, bounded shell growth turns this into

$$
\Delta I_k^{(\rho)}(W) = O(\rho^{2-\alpha}).
$$

### 5.2 Objective perturbation

Dropping far-field interference can only increase the surrogate rate, so

$$
0 \le \widetilde{R}_k^{(\rho)}(W) - R_k(W).
$$

Using the noise floor,

$$
\widetilde{R}_k^{(\rho)}(W) - R_k(W)
\le
\frac{\Delta I_k^{(\rho)}(W)}{\ln 2 \, \sigma_{\min}^2}.
$$

Therefore, uniformly over feasible `W`,

$$
0 \le \widetilde{F}^{(\rho)}(W) - F(W) \le K_F \rho^{2-\alpha}.
$$

### 5.3 Compare the two optimizers

Since `W_rho^*` maximizes the surrogate,

$$
F^\star = F(W^\star)
\le \widetilde{F}^{(\rho)}(W^\star)
\le \widetilde{F}^{(\rho)}(W_\rho^\star)
\le F(W_\rho^\star) + K_F \rho^{2-\alpha}.
$$

So

$$
0 \le F^\star - \widehat{F}_\rho^\star \le K_F \rho^{2-\alpha}.
$$

That is the theorem.

## 6. Important Nuance

If you define

$$
F_\rho^\star := \max_W F^{(\rho)}(W)
$$

using a surrogate that simply drops far-field interference, then `F_rho^*` is generally **optimistic**. In that formulation, the statement

$$
0 \le F^\star - F_\rho^\star
$$

does not hold in general.

To recover that more classical-looking theorem, you need a **conservative** localized model, for example by adding a residual term `eta_k^(rho)` that upper-bounds omitted interference in every denominator.

## 7. Does The Simulation Support The Intuition?

Yes, on the true objective.

### 7.1 Main locality study

From [results/raw/locality_scaling_summary.csv](/Users/yosia/Desktop/ideas/bs_coop/results/raw/locality_scaling_summary.csv), the 37-BS / 2-user-per-BS study gives:

| rho | mean weighted sum-rate | mean gap to global |
| --- | ---: | ---: |
| 1 | 76.70 | 18.70 |
| 2 | 89.66 | 5.74 |
| 3 | 93.38 | 2.02 |
| 4 | 94.73 | 0.67 |
| 5 | 95.16 | 0.24 |

So the true gap drops sharply as `rho` grows.

From [results/raw/locality_scaling_fit.csv](/Users/yosia/Desktop/ideas/bs_coop/results/raw/locality_scaling_fit.csv):

- empirical log-log slope: `-2.6345`
- reference exponent `2 - alpha` with `alpha = 3.6`: `-1.6`

This is consistent with the theorem. The theorem gives an upper-bound envelope, not an exact equality.

### 7.2 Robustness to path-loss exponent

From [results/raw/alpha_sweep_summary.csv](/Users/yosia/Desktop/ideas/bs_coop/results/raw/alpha_sweep_summary.csv), the mean gap-to-global curves for `alpha in {3.0, 3.6, 4.2}` all shrink rapidly with `rho`.

At `rho = 5`, the absolute mean gaps are:

- `alpha = 3.0`: `0.2695`
- `alpha = 3.6`: `0.1361`
- `alpha = 4.2`: `0.1064`

Using the corresponding mean global objective level, the relative gaps at `rho = 5` are approximately:

- `alpha = 3.0`: `0.35%`
- `alpha = 3.6`: `0.14%`
- `alpha = 4.2`: `0.09%`

From [results/raw/alpha_sweep_fit.csv](/Users/yosia/Desktop/ideas/bs_coop/results/raw/alpha_sweep_fit.csv), the empirical log-log slopes are:

- `alpha = 3.0`: `-2.3673`
- `alpha = 3.6`: `-2.9628`
- `alpha = 4.2`: `-3.1814`

This supports the qualitative theorem prediction that stronger spatial decay makes locality more effective.

## 8. Honest Caveat

The finite-iteration, nonconvex algorithm does **not** have to improve monotonically with `rho` on every single random instance.

What the current experiments support is the stronger and more useful claim:

- on averaged runs and larger layouts, the **true** gap to the global benchmark shrinks rapidly with `rho`
- stronger path-loss decay makes that shrinkage faster
- medium-sized neighborhoods already recover most of the global benefit

## 9. Bottom Line

Your core intuition is correct, but the theorem should be stated carefully:

- If the localized problem is optimistic, prove convergence for the **true objective attained by a localized optimizer**.
- If you want convergence of the localized surrogate optimum itself, use a conservative residual-interference correction.

For the current repository outputs, the evidence is favorable:

- increasing `rho` clearly drives performance toward the global benchmark
- the measured decay is at least as strong as the theoretical upper-bound trend
- larger `alpha` makes localized cooperation work even better
