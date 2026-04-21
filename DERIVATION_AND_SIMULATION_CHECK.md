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

This is the successor to the 2008 Hadisusanto-style decomposition:

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

The cleanest statement separates the exact finite-network identity from the asymptotic decay bound.

### 4.1 Theorem: Finite-Network Exact Equality

Let the set of base stations be finite and define the network diameter

$$
D_{\mathrm{net}} := \max_{b,c \in \mathcal{B}} d(b,c).
$$

Then for every `rho >= D_net`, the localized problem is exactly the global problem.

#### Proof

If `rho >= D_net`, then for every base station `b`,

$$
\mathcal{C}_b(\rho) = \mathcal{B}.
$$

Therefore no interference terms are omitted, so for every feasible beamformer collection `W`,

$$
\widetilde{F}^{(\rho)}(W) = F(W).
$$

Since the feasible set is also unchanged, the localized and global optimization problems are identical. Hence:

$$
\widehat{F}_\rho^\star = F^\star
\qquad \text{for all } \rho \ge D_{\mathrm{net}}.
$$

This is the rigorous version of the statement

> as local cooperation reaches the whole network, local cooperation equals global cooperation.

Once every neighborhood contains every base station, the localized model has the same variables, same interference terms, and same feasible set as the global model. So the two optimization problems are identical.

### 4.2 Corollary: `rho -> infinity` Implies Local = Global

Because the network is finite, `D_net < infinity`. Therefore there exists a finite threshold `rho_0 = D_net` such that

$$
\widehat{F}_\rho^\star = F^\star
\qquad \text{for all } \rho \ge \rho_0.
$$

Hence

$$
\lim_{\rho \to \infty} \widehat{F}_\rho^\star = F^\star.
$$

So on a finite network, exact equality happens before the limit: once `rho` reaches the network diameter, local cooperation and global cooperation are literally the same thing.

### 4.3 Theorem: Asymptotic Decay Bound For Expanding Networks

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

### 4.4 Extension To Fully MIMO Multi-Stream Users

The same logic carries over to the fully MIMO model used in the 2008 paper. Let user `k` have receive-channel blocks `H_{k,b}` and let BS `b` transmit to user `u` with a beamforming matrix `V_{b,u}` carrying `d_u` streams.

The omitted far-field term is now an interference covariance tail,

$$
\Delta \Sigma_k^{(\rho)}(V)
=
\sum_{c \notin \mathcal{C}_{b(k)}(\rho)}
\sum_u
H_{k,c} V_{c,u} V_{c,u}^H H_{k,c}^H.
$$

If the channel matrices satisfy the same spatial decay bound in Frobenius norm and the feasible beamformers are uniformly bounded, then

$$
\|\Delta \Sigma_k^{(\rho)}(V)\|_2 = O(\rho^{2-\alpha}).
$$

The user-rate expression in the MIMO case,

$$
R_k(V) = \log_2 \det \left(I + S_k(V)\Sigma_k(V)^{-1}\right),
$$

is still a continuous monotone function of the interference covariance. So the finite-network equality result and the asymptotic `O(\rho^{2-\alpha})` gap bound both extend to the multi-stream MIMO setting.

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

| rho | mean weighted sum-rate | mean gap to full-neighborhood |
| --- | ---: | ---: |
| 1 | 118.05 | 11.90 |
| 2 | 127.00 | 2.95 |
| 3 | 128.61 | 1.34 |
| 4 | 129.48 | 0.47 |
| 5 | 129.75 | 0.20 |

So the true gap drops sharply as `rho` grows.

From [results/raw/locality_scaling_fit.csv](/Users/yosia/Desktop/ideas/bs_coop/results/raw/locality_scaling_fit.csv):

- empirical log-log slope: `-2.4638`
- reference exponent `2 - alpha` with `alpha = 3.6`: `-1.6`

This is consistent with the theorem. The theorem gives an upper-bound envelope, not an exact equality.

### 7.2 Robustness to path-loss exponent

From [results/raw/alpha_sweep_summary.csv](/Users/yosia/Desktop/ideas/bs_coop/results/raw/alpha_sweep_summary.csv), the gap-to-full-neighborhood curves for `alpha in {3.0, 3.6, 4.2}` still shrink rapidly with `rho`.

At `rho = 5`, the absolute gaps to the full-neighborhood reference are:

- `alpha = 3.0`: `0.3013`
- `alpha = 3.6`: `0.1026`
- `alpha = 4.2`: `0.1400`

Using the corresponding full-neighborhood objective levels, the relative gaps at `rho = 5` are approximately:

- `alpha = 3.0`: `0.29%`
- `alpha = 3.6`: `0.08%`
- `alpha = 4.2`: `0.09%`

From [results/raw/alpha_sweep_fit.csv](/Users/yosia/Desktop/ideas/bs_coop/results/raw/alpha_sweep_fit.csv), the empirical log-log slopes are:

- `alpha = 3.0`: `-2.0845`
- `alpha = 3.6`: `-2.7812`
- `alpha = 4.2`: `-2.8005`

This supports the qualitative theorem prediction that stronger spatial decay makes locality more effective.

## 8. Honest Caveat

The finite-iteration, nonconvex algorithm does **not** have to improve monotonically with `rho` on every single random instance.

The current multi-stream MIMO alpha sweep also uses only `1` Monte Carlo trial per `alpha` in the default configuration to keep the full repository run practical, so those alpha-slope numbers should be treated as exploratory rather than final.

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
