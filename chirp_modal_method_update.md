# Chirp-Modal Latent Diffusion — Method & Theory (Direction 1)

*Drop-in method section + proof sketches for Theorems A–D (with B′, the certified bounded-growth variant). Notation is self-contained; rename the model freely (placeholder: **CMD**, Chirp-Modal Dynamics).*

---

## 4. Method

### 4.1 Setup and notation

We forecast/impute irregular multivariate time series by generating a low-dimensional **latent trajectory**
$z_0(\cdot):[0,H]\to\mathbb{R}^{d_z}$ over a horizon of length $H$, in relative time $\tilde t=t^{q}-t^{q}_1$ (so $\tilde t=0$ is the forecast start; for imputation $\tilde t$ may be negative). Following the latent-diffusion construction, a pretrained VAE encodes the target window to $z_0$ and decodes back; a conditional denoiser $p_\theta(z\mid E)$ operates in latent space, where $E=S_\phi(H_{t_i})$ is the gap-aware history summary. The forward process is the standard DDPM noising of $z_0$ indexed by diffusion step $\tau$; sampling is DDIM with $x_0$-parameterization. Our contribution replaces the denoiser's **dynamical core**: instead of predicting a *time-invariant* modal (LTI) system and patching its error with a residual MLP, the denoiser predicts a *time-varying* modal system whose trajectory is still available in closed form, is stable by construction, and admits analytic predictive uncertainty.

Throughout, $J=\begin{psmallmatrix}0&-1\\1&0\end{psmallmatrix}$, $\mathrm{Rot}(\theta)=\begin{psmallmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{psmallmatrix}=\cos\theta\,I_2+\sin\theta\,J$, and $\|\cdot\|$ is the Euclidean / spectral norm.

### 4.2 Time-varying modal generator (chirp-modal dynamics)

We use $K$ conjugate-mode blocks. For mode $k$ we predict two **positive scalar functions of relative time** — an instantaneous decay $\rho_k(\tilde t)>0$ and an instantaneous angular frequency $\omega_k(\tilde t)>0$ — and assemble the **rotation–scaling generator**

$$
A_k(\tilde t)\;=\;-\rho_k(\tilde t)\,I_2+\omega_k(\tilde t)\,J\;=\;\begin{pmatrix}-\rho_k(\tilde t)&-\omega_k(\tilde t)\\[2pt]\omega_k(\tilde t)&-\rho_k(\tilde t)\end{pmatrix}.\tag{1}
$$

The per-mode latent state $\xi_k\in\mathbb{R}^2$ evolves by the **linear time-varying (LTV)** ODE

$$
\dot\xi_k(\tilde t)=A_k(\tilde t)\,\xi_k(\tilde t),\qquad \xi_k(0)=\xi_k^0,\tag{2}
$$

and the latent readout is a sum of mode contributions, $z_0(\tilde t)=\sum_{k=1}^{K}C_k\,\xi_k(\tilde t)$ with $C_k\in\mathbb{R}^{d_z\times2}$. Letting the denoiser output **constant latent residues** $c_k,b_k\in\mathbb{R}^{d_z}$ (the cosine/sine components absorbing $C_k\xi_k^0$, see Thm A), the model class is fully specified by the per-mode tuple $\{(\rho_k(\cdot),\omega_k(\cdot),c_k,b_k)\}_{k=1}^K$.

**Why time-varying poles.** The LLapDiff core fixes $\rho_k,\omega_k$ over the entire window, i.e. it is **LTI**, whose only stable closed-form propagator is $e^{At}$. A single set of poles cannot express frequency drift, amplitude-dependent period, or regime change, so LLapDiff adds an unconstrained correction $\hat z_0\!\leftarrow\!\hat z_0+\mathrm{MLP}(\hat z_0)$ that carries **no stability certificate**. (One caveat we state precisely rather than gloss: with $\rho_k>0$ pointwise, each mode's envelope $e^{-\bar\rho_k}$ is monotone nonincreasing, so *per-mode transient amplitude growth* is **not** unlocked by time variation alone — it requires the certified bounded-growth relaxation of §4.4, Theorem B′.) Making the poles functions of $\tilde t$ removes that limitation — *provided* the resulting LTV system stays closed-form. Theorem A shows the normal form (1) is exactly the parameterization for which this holds.

**Parameterization (positivity + exact integration).** Define the integrated poles $\bar\rho_k(\tilde t)=\int_0^{\tilde t}\rho_k(s)\,ds$ and $\bar\omega_k(\tilde t)=\int_0^{\tilde t}\omega_k(s)\,ds$. We use either:

- **(P-exact)** a nonnegative basis with closed-form antiderivatives, $\rho_k(\tilde t)=\rho_{\min}+\sum_{m}(a^{\rho}_{km})^2\,\phi_m(\tilde t)$ (e.g. squared random-Fourier features or RBFs), so $\bar\rho_k(\tilde t)=\rho_{\min}\tilde t+\sum_m (a^{\rho}_{km})^2\,\Phi_m(\tilde t)$ is **analytic and parallel** over all query times; analogously for $\omega_k$; or
- **(P-mono)** a monotone network for $\bar\rho_k,\bar\omega_k$ directly (integral of a softplus head), whose derivatives give positive instantaneous poles by construction; or
- **(P-grid)** predict $\rho_k,\omega_k$ pointwise with a softplus head and integrate by cumulative trapezoid on the (irregular) query grid.

All three guarantee $\rho_k,\omega_k>0$ and need **no ODE solver**; (P-exact) and (P-mono) make the integrals exact.

### 4.3 Solver-free synthesis: the chirp-modal closed form

By Theorem A, the state transition of (2) collapses to an ordinary matrix exponential, giving the **chirp-modal synthesizer**

$$
\boxed{\;\hat z_0(\tilde t)=\sum_{k=1}^{K}e^{-\bar\rho_k(\tilde t)}\Big[\,c_k\cos\bar\omega_k(\tilde t)+b_k\sin\bar\omega_k(\tilde t)\,\Big]\;}\tag{3}
$$

This is a sum of **time-warped (chirped) damped sinusoids**: the amplitude envelope $e^{-\bar\rho_k}$ and the phase $\bar\omega_k$ vary across the window, so each mode can decelerate, accelerate, or change damping. Setting $\rho_k,\omega_k$ constant recovers $\bar\rho_k=\rho_k\tilde t$, $\bar\omega_k=\omega_k\tilde t$ and reduces (3) exactly to the LLapDiff fixed-pole expansion — **(3) is a strict generalization.** Like the LTI form, (3) is evaluated at all query times $\{\tilde t_r\}$ in a single parallel pass; unlike it, **no residual MLP is required** (§4.4, and ablation).

### 4.4 Stability by construction (no residual correction)

Because each transition operator is an orthogonal rotation scaled by $e^{-\bar\rho_k}$, positivity of the instantaneous decay yields a uniform contraction (Theorem B):

$$
\|\hat z_0(\tilde t)\|\;\le\;\sum_{k=1}^K e^{-\bar\rho_k(\tilde t)}\sqrt{\|c_k\|^2+\|b_k\|^2}\;\le\;e^{-\rho_{\min}\tilde t}\sum_{k=1}^K\sqrt{\|c_k\|^2+\|b_k\|^2}.\tag{4}
$$

The bound is **independent of the horizon $H$** and decays at least geometrically, so the predicted trajectory cannot diverge — the property LLapDiff sought but lost when it added an uncertified MLP. We therefore **drop the residual correction**. If extra local expressiveness is desired, one may add a *certified* correction: any term whose own envelope is dominated by $e^{-\rho_{\min}\tilde t}$ (e.g. a residual whose output is gated by $e^{-\rho_{\min}\tilde t}$ or a 1-Lipschitz map composed with the decaying modal state) preserves (4); an unconstrained $\mathrm{MLP}(\hat z_0)$ does not.

**Certified transient growth (bounded-growth variant).** Strict positivity $\rho_k(\tilde t)\ge\rho_{\min}>0$ makes every per-mode envelope monotone nonincreasing: within-window amplitude growth (a rising transient, a building storm, a rally) is representable only through cross-mode cancellation. To admit genuine growth *without surrendering the certificate*, we relax the pointwise constraint to a **budgeted excursion** on the integrated decay. Parameterize

$$
\bar\rho_k(\tilde t)\;=\;\rho_{\min}\tilde t\;+\;h_k(\tilde t)\;-\;\gamma_k(\tilde t),\qquad h_k\ \text{nondecreasing},\ h_k(0)=0,\qquad \gamma_k:[0,H]\to[0,c_g],\ \gamma_k(0)=0,\tag{4$'$a}
$$

with a global (or per-mode) **growth budget** $c_g\ge0$. The instantaneous decay $\rho_k=\rho_{\min}+h_k'-\gamma_k'$ may now be **negative** wherever $\gamma_k'>\rho_{\min}+h_k'$ — the envelope genuinely grows there — but the total multiplicative growth over any subinterval is capped at $e^{c_g}$, and Theorem B′ gives the horizon-independent certificate

$$
\|\hat z_0(\tilde t)\|\;\le\;e^{c_g}\,e^{-\rho_{\min}\tilde t}\sum_{k=1}^K\sqrt{\|c_k\|^2+\|b_k\|^2}.\tag{4$'$}
$$

This is exactly exponential stability *with overshoot constant* $M=e^{c_g}$ — the standard notion $\|\Phi(t)\|\le Me^{-\lambda t}$ from control, where for non-normal LTI systems the overshoot is an uncontrolled artifact; here it is a **learned, explicitly capped** quantity, and $c_g=0$ recovers Theorem B verbatim. Theorem A is untouched (commutativity is sign-free), and Theorem C degrades gracefully by the factor $e^{2c_g}$ (Thm B′(iv)). Implementation under (P-exact): $\gamma_k(\tilde t)=c_g\big[\sigma(g_k(\tilde t))-\sigma(g_k(0))\big]$ with $g_k$ a basis expansion and $\sigma$ the logistic function keeps $\gamma_k\le c_g$, anchors $\gamma_k(0)=0$ (values $\gamma_k<0$ only add decay and are harmless), and has closed-form derivative for the instantaneous pole.

*Remark (scope of the certificate).* Theorems B/B′ certify the forward window $\tilde t\in[0,H]$. For imputation queries the raw offset $t^q_r-t^q_1$ could be negative under a mid-window anchor, where $e^{-\bar\rho_k}$ *grows* backward in time; we therefore anchor $\tilde t$ at the **earliest** query time so that all evaluations lie in $[0,H]$ and the certificate applies to forecasting and imputation alike.

### 4.5 Stochastic dynamics and closed-form predictive uncertainty

To obtain physically grounded uncertainty without sampling, we lift each mode to a linear SDE,

$$
d\xi_k=A_k(\tilde t)\,\xi_k\,d\tilde t+\Sigma_k(\tilde t)\,dW_k,\qquad \xi_k(0)\sim\mathcal N(\mu_k^0,P_k^0),\tag{5}
$$

with $Q_k(\tilde t):=\Sigma_k\Sigma_k^\top$. The conditional mean reproduces (3); the covariance obeys a differential **Lyapunov** equation. Under the natural isotropic-in-block choice $Q_k=q_k(\tilde t)I_2$ and $P_k^0=p_k^0 I_2$, Theorem C gives the **closed-form moments**

$$
m_k(\tilde t)=\Phi_k(\tilde t)\mu_k^0,\qquad
P_k(\tilde t)=\Big[e^{-2\bar\rho_k(\tilde t)}p_k^0+\underbrace{\int_0^{\tilde t}e^{-2(\bar\rho_k(\tilde t)-\bar\rho_k(s))}q_k(s)\,ds}_{v_k(\tilde t)}\Big]I_2,\tag{6}
$$

and, for independent modes, an analytic predictive law in latent space,

$$
z_0(\tilde t)\sim\mathcal N\!\Big(\textstyle\sum_k C_k m_k(\tilde t),\ \sum_k\big[e^{-2\bar\rho_k(\tilde t)}p_k^0+v_k(\tilde t)\big]\,C_kC_k^\top\Big).\tag{7}
$$

The scalar $v_k$ is a 1-D integral evaluated by a **solver-free quadrature** over the query grid: an exponential-integrator recurrence $v_r=e^{-2\Delta\bar\rho_r}v_{r-1}+q\,\Delta t_r\,(1-e^{-2\Delta\bar\rho_r})/(2\Delta\bar\rho_r)$ whose exponents are all $\le 0$ (overflow-free) and which is *exact* for piecewise-constant poles. (Under (P-exact) the integrand $e^{2\bar\rho_k(s)}q_k(s)$ contains the exponential of a trigonometric polynomial and admits no elementary antiderivative, so — unlike the pole integrals — $v_k$ has a closed form only in the constant-pole case, where it equals $q_k(1-e^{-2\rho_k\tilde t})/(2\rho_k)\to q_k/(2\rho_k)$, the algebraic-Lyapunov steady-state variance; the quadrature reproduces this to float precision.) Equation (7) yields **predictive intervals at every query time in one forward pass** — replacing LLapDiff's many-step sampled UQ — and is propagated to data space through the decoder (exactly if linear; delta method otherwise). Cross-mode correlation is handled by a joint block-Lyapunov solve, still closed-form in the eigenbasis of the stacked generator.

### 4.6 Conditioning, training, inference

The denoiser is the pair (predictor $L_\theta$, synthesizer $L_\theta^{+}$). Given $(z_\tau,\tau,E)$, $L_\theta$ outputs the pole-function coefficients and residues $\{(\rho_k(\cdot),\omega_k(\cdot),c_k,b_k)\}$ (and, in the stochastic variant, $\{q_k(\cdot),p_k^0\}$); $L_\theta^{+}$ evaluates (3) (and (7)) at the query times. Irregular timestamps enter **only** through $\tilde t_r$ in the closed form; gap statistics enter through $E$, consistent with the renewal-averaging motivation, which Theorem D generalizes to time-varying poles (§4.7). Training minimizes the $x_0$ reconstruction loss $\mathbb E\|z_0-\hat z_0(z_\tau,\tau,E)\|^2$ (optionally a Gaussian NLL using (7) for calibrated UQ) under classifier-free guidance; inference is deterministic DDIM. The poles $\{\rho_k(\cdot),\omega_k(\cdot)\}$ are predicted **once per denoising step** and reused across the synthesizer's parallel time evaluation.

### 4.7 Renewal averaging with time-varying poles

LLapDiff's renewal analysis maps a *constant* pole $s_k=-\rho_k+i\omega_k$ under i.i.d. gaps $\Delta$ to the effective event-domain log-pole $\bar s_k=\log\mathbb E[e^{s_k\Delta}]\approx s_k\mathbb E[\Delta]+\tfrac12 s_k^2\operatorname{Var}(\Delta)$. Theorem D extends this to pole **functions**. With $s_k(t)=-\rho_k(t)+i\omega_k(t)$, the exact per-event map is $\zeta_{j+1}=\exp\!\big(\int_{t_j}^{t_{j+1}}s_k(\tau)\,d\tau\big)\zeta_j$ — and it is Theorem A that makes this event map *closed-form* in real coordinates, since for a general LTV realization the same map is a time-ordered exponential. Three consequences (proofs in the appendix):

1. **Stability is preserved — pathwise, not just in mean.** Because the envelope $e^{-\bar\rho_k}$ is deterministic given the gaps, $|\zeta_{j+1}|\le e^{-\rho_{\min}\Delta_{j+1}}|\zeta_j|$ holds *per sample path*, strengthening the constant-pole mean-stability result; under the growth budget the same holds up to the capped factor $e^{c_g}$.
2. **The effective log-pole acquires a drift term.** The conditional multiplier $\lambda_k(t)=\mathbb E\big[e^{\int_t^{t+\Delta}s_k}\big]$ satisfies, to second order,
$$
\bar s_k(t)\;=\;\log\lambda_k(t)\;\approx\;s_k(t)\,\mathbb E[\Delta]\;+\;\tfrac12\,s_k(t)^2\operatorname{Var}(\Delta)\;+\;\tfrac12\,s_k'(t)\,\mathbb E[\Delta^2],\tag{8}
$$
recovering LLapDiff's expansion when $s_k'\equiv0$. The new term couples **pole drift to the second raw moment of the gaps**: increasing damping ($\rho_k'>0$) contributes extra event-domain decay $\propto\mathbb E[\Delta^2]$, and an up-chirp ($\omega_k'>0$) inflates the per-event phase increment $\propto\mathbb E[\Delta^2]$.
3. **Architectural corollary.** In the event domain, intrinsic poles, pole drift, and gap statistics are three-way entangled by (8); disentangling $(\rho_k(\cdot),\omega_k(\cdot))$ requires conditioning on at least the first two gap moments — precisely what the gap-aware temporal tokens supply. This upgrades LLapDiff's motivation to the time-varying setting and yields a *testable prediction*: learned pole functions should remain invariant across sampling-gap regimes while the event-domain multipliers shift (an experiment analogous to LLapDiff's induced-missingness stress test, evaluated on the pole trajectories).

---

## Appendix — Theorems and proof sketches

We collect the standing regularity assumption and the three results. Proofs are sketched at the level expected in the main appendix; full details (Carathéodory existence, dominated convergence in the integrals) are routine.

**Assumption R.** For each $k$, $\rho_k,\omega_k\in L^1([0,H])$ with $\rho_k\ge 0$ a.e.; in Thm B additionally $\rho_k\ge\rho_{\min}>0$ a.e. Then $\bar\rho_k,\bar\omega_k$ are absolutely continuous and (2),(5) have unique (strong) solutions.

**Assumption R$^+$ (bounded growth).** For each $k$, $\omega_k\in L^1([0,H])$ and $\rho_k\in L^1([0,H])$ is *signed*, with integrated decay of the form (4$'$a): $\bar\rho_k(t)=\rho_{\min}t+h_k(t)-\gamma_k(t)$, where $h_k$ is nondecreasing with $h_k(0)=0$ and $\gamma_k:[0,H]\to[0,c_g]$ with $\gamma_k(0)=0$, for constants $\rho_{\min}>0$, $c_g\ge0$. (Assumption R with $\rho_k\ge\rho_{\min}$ is the case $c_g=0$.) Note Theorem A requires only $\rho_k,\omega_k\in L^1$ and is indifferent to the sign of $\rho_k$; it holds verbatim under R$^+$.

**Assumption G (gaps).** Event times $t_0=0<t_1<t_2<\cdots$ with gaps $\Delta_j:=t_j-t_{j-1}\ge0$ i.i.d. and independent of the mode state; for the expansion in Thm D(d), additionally $s_k\in C^1$ in a neighborhood of the evaluation time and $\mathbb E[\Delta^3]<\infty$. Nonstationary or history-dependent gaps are handled by conditioning (Thm D, final remark).

### Theorem A (Exact integrability of the rotation–scaling normal form)

*Under Assumption R, the family $\{A_k(t)\}_{t\in[0,H]}$ in (1) is commutative, and the state transition of (2) is*
$$
\Phi_k(t)=\exp\!\Big(\!\int_0^t A_k(s)\,ds\Big)=e^{-\bar\rho_k(t)}\,\mathrm{Rot}\big(\bar\omega_k(t)\big),\qquad t\in[0,H].
$$
*Consequently the latent readout admits the closed form (3), with $c_k,b_k$ linear in $(C_k,\xi_k^0)$.*

**Proof sketch.**
*(i) Commutativity.* Let $\mathcal A=\{aI_2+bJ:a,b\in\mathbb R\}$. Since $I_2$ is central and, using $J^2=-I_2$,
$$(aI+bJ)(a'I+b'J)=(aa'-bb')I+(ab'+a'b)J,$$
which is symmetric under swapping the two factors, $\mathcal A$ is a commutative algebra (indeed $\mathcal A\cong\mathbb C$ via $aI+bJ\mapsto a+bi$). Each $A_k(t)=-\rho_k(t)I+\omega_k(t)J\in\mathcal A$, so $A_k(t)A_k(t')=A_k(t')A_k(t)$.

*(ii) Exponential of the integral solves the IVP.* $\mathcal A$ is a closed linear subspace, so $M(t):=\int_0^t A_k(s)\,ds\in\mathcal A$; hence $M(t)$ and $\dot M(t)=A_k(t)$ commute. The Duhamel formula $\frac{d}{dt}e^{M(t)}=\int_0^1 e^{sM}\dot M\,e^{(1-s)M}\,ds$ collapses, because commuting factors recombine, to $\dot M(t)\,e^{M(t)}=A_k(t)\,e^{M(t)}$. With $e^{M(0)}=I_2$ and uniqueness of linear ODE solutions, $\Phi_k=e^{M}$.

*(iii) Evaluating the exponential.* $M(t)=-\bar\rho_k(t)I+\bar\omega_k(t)J\mapsto-\bar\rho_k(t)+i\bar\omega_k(t)$ under $\mathcal A\cong\mathbb C$, and matrix-exp corresponds to complex-exp: $e^{-\bar\rho_k+i\bar\omega_k}=e^{-\bar\rho_k}(\cos\bar\omega_k+i\sin\bar\omega_k)$, mapping back to $e^{-\bar\rho_k}(\cos\bar\omega_k\,I+\sin\bar\omega_k\,J)=e^{-\bar\rho_k}\mathrm{Rot}(\bar\omega_k)$.

*(iv) Readout.* With $\xi_k(t)=\Phi_k(t)\xi_k^0$ and $C_k=[\gamma_k\ \delta_k]$,
$$C_k\Phi_k(t)\xi_k^0=e^{-\bar\rho_k}\big[(\gamma_k\xi^0_1+\delta_k\xi^0_2)\cos\bar\omega_k+(\delta_k\xi^0_1-\gamma_k\xi^0_2)\sin\bar\omega_k\big],$$
so defining $c_k:=\gamma_k\xi^0_1+\delta_k\xi^0_2$ and $b_k:=\delta_k\xi^0_1-\gamma_k\xi^0_2$ and summing over $k$ gives (3). $\qquad\blacksquare$

#### Proposition A.1 (Necessity of the normal form: companion realizations are not integrable)

*The collapse in Theorem A is special to the rotation–scaling form. For the companion realization of a varying-frequency oscillator, $\tilde A(t)=\begin{psmallmatrix}0&1\\-\omega^2(t)&0\end{psmallmatrix}$, one has*
$$[\tilde A(t),\tilde A(t')]=\big(\omega^2(t)-\omega^2(t')\big)\begin{pmatrix}1&0\\0&-1\end{pmatrix}\neq 0\quad\text{whenever }\omega(t)\neq\omega(t'),$$
*so $\Phi\neq\exp(\int\tilde A)$ in general and no elementary closed form exists; the leading-order (WKB/adiabatic) solution of $\ddot y+\omega^2(t)y=0$ is $y(t)\approx\omega(t)^{-1/2}\cos\!\int_0^t\omega$, carrying the amplitude factor $\omega^{-1/2}$ absent from a naive $\cos\!\int\omega$.*

**Proof sketch.** Direct $2\times2$ multiplication gives the commutator. Non-commutativity invalidates the Duhamel collapse used in Thm A(ii). The WKB statement is the standard Liouville–Green approximation: substituting $y=\exp(\int(\,\cdot\,))$ and balancing orders yields slowly-varying amplitude $\propto\omega^{-1/2}$. $\qquad\blacksquare$

*Remark.* Proposition A.1 is the design justification for (1): among $2\times2$ realizations of a conjugate pole pair, the rotation–scaling form is the one whose time-varying version stays exactly solvable, because it lives in the commutative algebra $\mathcal A\cong\mathbb C$. This is why we predict $(\rho_k,\omega_k)$ in normal form rather than, e.g., companion coefficients.

### Theorem B (Stability / contraction by construction)

*Under Assumption R with $\rho_k\ge\rho_{\min}>0$: (i) each transition is a strict contraction, $\|\Phi_k(t)\|=e^{-\bar\rho_k(t)}\le e^{-\rho_{\min}t}<1$ for $t>0$; (ii) the synthesized trajectory obeys (4); (iii) hence $\sup_{t\in[0,H]}\|\hat z_0(t)\|$ is bounded by an $H$-independent constant and $\hat z_0(t)\to 0$ as $t\to\infty$, with no residual term.*

**Proof sketch.**
*(i)* By Thm A, $\Phi_k(t)=e^{-\bar\rho_k(t)}\mathrm{Rot}(\bar\omega_k(t))$; rotations are orthogonal so $\|\mathrm{Rot}\|=1$, giving $\|\Phi_k(t)\|=e^{-\bar\rho_k(t)}$. Positivity gives $\bar\rho_k(t)=\int_0^t\rho_k\ge\rho_{\min}t$.

*(ii)* Triangle inequality on (3): $\|\hat z_0(t)\|\le\sum_k e^{-\bar\rho_k(t)}\|c_k\cos\bar\omega_k+b_k\sin\bar\omega_k\|$. For the amplitude, $\|c\cos\theta+b\sin\theta\|^2$ equals $u^\top G u$ with $u=(\cos\theta,\sin\theta)^\top$ and Gram $G=\begin{psmallmatrix}\|c\|^2&\langle c,b\rangle\\\langle c,b\rangle&\|b\|^2\end{psmallmatrix}$, hence $\le\lambda_{\max}(G)\le\operatorname{tr}G=\|c\|^2+\|b\|^2$. Combining with (i) yields (4).

*(iii)* The right-hand side of (4) is $e^{-\rho_{\min}t}\sum_k\sqrt{\|c_k\|^2+\|b_k\|^2}$; the residues are finite network outputs, so the bound is uniform on $[0,H]$, independent of $H$, and $\to 0$. No further term enters because the entire output is the modal sum. $\qquad\blacksquare$

*Remark (contrast with LLapDiff).* The map $\hat z_0\mapsto\hat z_0+\mathrm{MLP}(\hat z_0)$ has no a priori norm or Lipschitz bound, so a certificate on the modal part does not transfer to the output; the residue-gated / Lipschitz corrections noted in §4.4 keep an envelope $\preceq e^{-\rho_{\min}t}$ and so preserve (4). **Precision for the released implementation:** in LLapDiff's code the genuinely uncertified term is the *output head* $\hat z_0\mapsto s\,\hat z_0+\mathrm{Linear}(\mathrm{LayerNorm}(\hat z_0))$ — the LayerNorm re-normalizes its input to unit scale as the modal envelope decays, so the head term does **not** decay and the output tends to a nonzero floor (violating (iii)). The *inner* synthesis MLP is spectral-normalized (1-Lipschitz, zero-initialized), i.e. by itself close to a certified correction in the sense of §4.4. The component our method deletes is therefore the head (while retaining its learnable output scaling $s$, clamped to $|s|\le1$ so it only rescales the bound constant); claims should name the head, not "the MLP", or a code inspection will find the certificate argument misdirected.

### Theorem B′ (Certified transient growth under a budget)

*Under Assumption R$^+$: (i) for all $0\le s\le t\le H$, the two-time transition satisfies*
$$
\|\Phi_k(t,s)\|\;=\;e^{-(\bar\rho_k(t)-\bar\rho_k(s))}\;\le\;e^{c_g}\,e^{-\rho_{\min}(t-s)},
$$
*in particular $\|\Phi_k(t)\|\le e^{c_g}e^{-\rho_{\min}t}$; (ii) the synthesized trajectory obeys the horizon-independent bound (4$'$), and $\hat z_0(t)\to0$ as $t\to\infty$; (iii) the instantaneous decay $\rho_k=\rho_{\min}+h_k'-\gamma_k'$ may be negative on sets of positive measure — the envelope genuinely grows there — but the multiplicative growth over any subinterval $[s,t]$ never exceeds $e^{c_g}$; (iv) Theorem C continues to hold with the variance bound $v_k(t)\le e^{2c_g}\int_0^t e^{-2\rho_{\min}(t-s)}q_k(s)\,ds\le e^{2c_g}\,\bar q_k\,(1-e^{-2\rho_{\min}t})/(2\rho_{\min})$ for $q_k\le\bar q_k$. Setting $c_g=0$ recovers Theorem B.*

**Proof sketch.**
*(i)* Theorem A holds under R$^+$ (its hypotheses are only $L^1$ integrability; commutativity of $\{aI+bJ\}$ is indifferent to signs), so $\Phi_k(t,s)=e^{-(\bar\rho_k(t)-\bar\rho_k(s))}\mathrm{Rot}(\bar\omega_k(t)-\bar\omega_k(s))$ and $\|\Phi_k(t,s)\|=e^{-(\bar\rho_k(t)-\bar\rho_k(s))}$. From (4$'$a),
$$\bar\rho_k(t)-\bar\rho_k(s)=\rho_{\min}(t-s)+\underbrace{h_k(t)-h_k(s)}_{\ge0}-\underbrace{(\gamma_k(t)-\gamma_k(s))}_{\le c_g}\;\ge\;\rho_{\min}(t-s)-c_g,$$
using monotonicity of $h_k$ and $\gamma_k(t)\le c_g$, $\gamma_k(s)\ge0$.

*(ii)* Identical Gram-matrix argument as Thm B(ii), with the envelope bound of (i) at $s=0$ (where $\gamma_k(0)=0$ gives $\bar\rho_k(t)\ge\rho_{\min}t-c_g$ directly).

*(iii)* $\rho_k<0$ iff $\gamma_k'>\rho_{\min}+h_k'$, which the parameterization permits pointwise; the growth cap is (i) rearranged: $e^{-(\bar\rho_k(t)-\bar\rho_k(s))}\le e^{c_g}e^{-\rho_{\min}(t-s)}\le e^{c_g}$.

*(iv)* Insert the two-time bound of (i) into the variance integral of Thm C(b): $\Phi_k(t,s)Q_k\Phi_k(t,s)^\top=q_k(s)e^{-2(\bar\rho_k(t)-\bar\rho_k(s))}I_2\preceq q_k(s)e^{2c_g}e^{-2\rho_{\min}(t-s)}I_2$; integrate. $\qquad\blacksquare$

*Remark (control-theoretic reading).* (i) is exponential stability with **overshoot constant** $M=e^{c_g}$, the classical notion $\|\Phi(t)\|\le Me^{-\lambda t}$. For non-normal LTI systems the overshoot is an emergent, uncontrolled artifact of non-normality; the normal-form blocks here are normal matrices, so *no* overshoot exists at $c_g=0$ — the budget re-introduces it as a **learned, explicitly capped design parameter**. This answers the expressiveness objection ("strictly positive damping forbids rising transients") without abandoning the certificate.

### Theorem C (Closed-form moment propagation / analytic UQ)

*Consider the linear SDE (5) under Assumption R. (a) The mean solves $\dot m_k=A_k m_k$, so $m_k(t)=\Phi_k(t)\mu_k^0$ (Thm A). (b) The covariance solves the differential Lyapunov equation*
$$
\dot P_k=A_k(t)P_k+P_kA_k(t)^\top+Q_k(t),\qquad P_k(0)=P_k^0,
$$
*with solution $P_k(t)=\Phi_k(t,0)P_k^0\Phi_k(t,0)^\top+\int_0^t\Phi_k(t,s)Q_k(s)\Phi_k(t,s)^\top\,ds$, where $\Phi_k(t,s)=e^{-(\bar\rho_k(t)-\bar\rho_k(s))}\mathrm{Rot}(\bar\omega_k(t)-\bar\omega_k(s))$. (c) If $Q_k=q_k(t)I_2$ and $P_k^0=p_k^0I_2$, the integral collapses to the scalar form (6); for independent modes the readout law is the closed-form Gaussian (7).*

**Proof sketch.**
*(a)* Take expectations in (5); the Itô integral is a martingale with zero mean, leaving $\dot m_k=A_k m_k$; apply Thm A.

*(b)* By Itô, $d(\xi_k\xi_k^\top)=(d\xi_k)\xi_k^\top+\xi_k(d\xi_k)^\top+Q_k\,dt$ (the quadratic-variation term is $\Sigma_k\Sigma_k^\top dt=Q_k\,dt$). Taking expectations gives $\frac{d}{dt}\mathbb E[\xi_k\xi_k^\top]=A_k\mathbb E[\xi_k\xi_k^\top]+\mathbb E[\xi_k\xi_k^\top]A_k^\top+Q_k$; subtracting $\frac{d}{dt}(m_km_k^\top)=A_km_km_k^\top+m_km_k^\top A_k^\top$ yields the Lyapunov equation for $P_k=\mathbb E[\xi_k\xi_k^\top]-m_km_k^\top$. Variation of constants with the transition $\Phi_k(t,s)$ — which equals $\exp\!\int_s^t A_k$ by the commutativity of Thm A — gives the integral solution.

*(c)* For $Q_k=q_k(s)I_2$, orthogonality of $\mathrm{Rot}$ gives $\Phi_k(t,s)Q_k\Phi_k(t,s)^\top=q_k(s)e^{-2(\bar\rho_k(t)-\bar\rho_k(s))}\,\mathrm{Rot}(\cdot)\mathrm{Rot}(\cdot)^\top=q_k(s)e^{-2(\bar\rho_k(t)-\bar\rho_k(s))}I_2$, so the integral is the scalar $v_k(t)I_2$; the initial term is $e^{-2\bar\rho_k(t)}p_k^0 I_2$ since $\mathrm{Rot}\,(p_k^0 I_2)\mathrm{Rot}^\top=p_k^0 I_2$. Hence (6). Independence of modes makes the stacked covariance block-diagonal, and $z_0=\sum_kC_k\xi_k$ (an affine map of jointly Gaussian states) is Gaussian with covariance $\sum_kC_kP_k(t)C_k^\top$, giving (7). The sanity check $\rho_k,q_k$ constant $\Rightarrow v_k=q_k(1-e^{-2\rho_k t})/(2\rho_k)$ recovers the algebraic-Lyapunov steady state $q_k/(2\rho_k)$. $\qquad\blacksquare$

*Remark (calibration and scope).* Under (5)–(7) the latent predictive law is **exactly** Gaussian, so nominal coverage is exact in latent space; observed miscalibration is attributable only to (i) decoder nonlinearity and (ii) the linear-latent approximation, both of which are quantified empirically (reliability diagrams / PIT). General (non-isotropic, correlated) modes retain a closed-form covariance via the block-Lyapunov solve $(\,I\otimes\mathcal A+\mathcal A\otimes I\,)\operatorname{vec}(P)$ in the eigenbasis of the stacked generator; we use the isotropic case in the main model for parallel efficiency.

### Theorem D (Renewal averaging for time-varying poles)

*Let $s_k(t)=-\rho_k(t)+i\omega_k(t)$ and let $\zeta^{(k)}(t)\in\mathbb C$ evolve as $\dot\zeta^{(k)}=s_k(t)\zeta^{(k)}$, sampled at event times satisfying Assumption G. Write $\zeta^{(k)}_j:=\zeta^{(k)}(t_j)$.*

*(a) **Exact event recursion.** $\zeta^{(k)}_{j+1}=\exp\!\big(\int_{t_j}^{t_{j+1}}s_k(\tau)\,d\tau\big)\,\zeta^{(k)}_j$, and in real coordinates $\xi^{(k)}_j=[\Re\zeta^{(k)}_j,\Im\zeta^{(k)}_j]^\top$,*
$$
\xi^{(k)}_{j+1}\;=\;e^{-(\bar\rho_k(t_{j+1})-\bar\rho_k(t_j))}\,\mathrm{Rot}\big(\bar\omega_k(t_{j+1})-\bar\omega_k(t_j)\big)\,\xi^{(k)}_j.
$$

*(b) **Pathwise contraction and mean stability.** Under Assumption R with $\rho_k\ge\rho_{\min}>0$: $|\zeta^{(k)}_{j+1}|\le e^{-\rho_{\min}\Delta_{j+1}}|\zeta^{(k)}_j|$ almost surely, hence*
$$
\mathbb E\big[|\zeta^{(k)}_j|\big]\;\le\;\big(\mathbb E[e^{-\rho_{\min}\Delta}]\big)^{j}\,|\zeta^{(k)}_0|,\qquad \big|\mathbb E[\zeta^{(k)}_j]\big|\le\mathbb E\big[|\zeta^{(k)}_j|\big],
$$
*with $\mathbb E[e^{-\rho_{\min}\Delta}]<1$ whenever $\mathbb P(\Delta>0)>0$. Under Assumption R$^+$ the same holds up to the capped factor: $|\zeta^{(k)}_j|=e^{-\bar\rho_k(t_j)}|\zeta^{(k)}_0|\le e^{c_g}e^{-\rho_{\min}t_j}|\zeta^{(k)}_0|$, whence $\mathbb E[|\zeta^{(k)}_j|]\le e^{c_g}\big(\mathbb E[e^{-\rho_{\min}\Delta}]\big)^j|\zeta^{(k)}_0|$; individual per-event multipliers may exceed $1$, but the running product is uniformly bounded by $e^{c_g}$.*

*(c) **Effective conditional log-pole and phase cancellation.** Define $\lambda_k(t):=\mathbb E\big[\exp\!\int_t^{t+\Delta}s_k(\tau)\,d\tau\big]$ and $\bar s_k(t):=\log\lambda_k(t)$ (principal branch). Under Assumption R, $\Re\,\bar s_k(t)\le0$ ($<0$ when $\rho_k\ge\rho_{\min}>0$ and $\mathbb P(\Delta>0)>0$). Moreover, with the exponentially tilted law $dQ_t/d\mathbb P\propto e^{-\int_t^{t+\Delta}\rho_k}$,*
$$
\lambda_k(t)\;=\;\mathbb E\Big[e^{-\int_t^{t+\Delta}\rho_k}\Big]\cdot \mathbb E_{Q_t}\Big[e^{\,i\int_t^{t+\Delta}\omega_k}\Big],\qquad \Big|\mathbb E_{Q_t}\big[e^{\,i\int\omega_k}\big]\Big|\le1,
$$
*with equality on the right only if the random phase increment $\int_t^{t+\Delta}\omega_k$ is a.s. constant modulo $2\pi$ — a condition that time variation of $\omega_k$ makes generically harder to satisfy, so oscillatory chirped modes typically receive strictly more event-domain contraction than their envelope alone provides.*

*(d) **Second-order expansion (generalizing the constant-pole map).** Under Assumption G with $s_k\in C^1$ near $t$,*
$$
\bar s_k(t)\;\approx\;s_k(t)\,\mathbb E[\Delta]\;+\;\tfrac12\,s_k(t)^2\,\operatorname{Var}(\Delta)\;+\;\tfrac12\,s_k'(t)\,\mathbb E[\Delta^2],\tag{8}
$$
*i.e., in real and imaginary parts,*
$$
\Re\,\bar s_k(t)\approx-\rho_k\mathbb E[\Delta]+\tfrac12(\rho_k^2-\omega_k^2)\operatorname{Var}(\Delta)-\tfrac12\rho_k'\,\mathbb E[\Delta^2],\qquad
\Im\,\bar s_k(t)\approx\omega_k\mathbb E[\Delta]-\rho_k\omega_k\operatorname{Var}(\Delta)+\tfrac12\omega_k'\,\mathbb E[\Delta^2].
$$
*Constant poles ($s_k'\equiv0$) recover LLapDiff's expansion exactly; the new term shows pole drift couples to the second raw moment $\mathbb E[\Delta^2]=\operatorname{Var}(\Delta)+\mathbb E[\Delta]^2$ of the gaps.*

*(e) **Nonstationary gaps.** If gaps are history-dependent, (a)–(c) hold with $\mathbb E$ replaced by $\mathbb E[\,\cdot\mid\mathcal F^{\mathrm{evt}}_j]$, i.e., $\mathbb E[\zeta^{(k)}_{j+1}\mid\mathcal F^{\mathrm{evt}}_j]=\lambda_k(t_j\mid\mathcal F^{\mathrm{evt}}_j)\,\zeta^{(k)}_j$ with the same modulus bounds.*

**Proof sketch.**
*(a)* Solve the scalar linear ODE exactly between events: $\zeta^{(k)}(t)=\exp\!\big(\int_{t_j}^{t}s_k\big)\zeta^{(k)}_j$ (Carathéodory, Assumption R/R$^+$). The real form is Theorem A applied on $[t_j,t_{j+1}]$: the family $\{A_k(t)\}\subset\mathcal A\cong\mathbb C$ commutes, so the event map is $\exp\!\int_{t_j}^{t_{j+1}}A_k$, evaluated as in Thm A(iii). *This is where Theorem A is load-bearing: for a non-commuting LTV realization (Prop. A.1) the event map is a time-ordered exponential with no closed form, and the renewal analysis below would not be available.*

*(b)* $|\zeta^{(k)}_{j+1}|=e^{-\int_{t_j}^{t_{j+1}}\rho_k}|\zeta^{(k)}_j|\le e^{-\rho_{\min}\Delta_{j+1}}|\zeta^{(k)}_j|$ pathwise, since $\rho_k\ge\rho_{\min}$. Telescoping and independence of the i.i.d. gaps give $\mathbb E\prod_{i\le j}e^{-\rho_{\min}\Delta_i}=(\mathbb E e^{-\rho_{\min}\Delta})^j$; $|\mathbb E(\cdot)|\le\mathbb E|\cdot|$ is the triangle inequality. Under R$^+$, telescope instead to $|\zeta^{(k)}_j|=e^{-\bar\rho_k(t_j)}|\zeta^{(k)}_0|$ and apply $\bar\rho_k(t)\ge\rho_{\min}t-c_g$ (Thm B′(i) at $s=0$). Note this is *stronger* than the constant-pole statement, which controlled only $|\mathbb E[\zeta_j]|$: here the envelope is deterministic given the gaps, so contraction holds per sample path.

*(c)* $|\lambda_k(t)|\le\mathbb E\big[|e^{\int s_k}|\big]=\mathbb E\big[e^{-\int_t^{t+\Delta}\rho_k}\big]\le\mathbb E[e^{-\rho_{\min}\Delta}]\le1$ (Jensen/triangle, as in the constant-pole case), so $\Re\,\bar s_k=\log|\lambda_k|\le0$. The tilting factorization is the time-varying analogue of the constant-pole argument: $e^{-\int\rho_k}\ge0$ and $\mathbb E[e^{-\int\rho_k}]>0$ make $Q_t$ a valid law; split $e^{\int s_k}=e^{-\int\rho_k}e^{i\int\omega_k}$ and normalize. The modulus of a characteristic-function-type average is $\le1$ with the stated equality condition.

*(d)* Pathwise Taylor expansion of the integral at $t$: $\int_t^{t+\Delta}s_k(\tau)\,d\tau=s_k(t)\Delta+\tfrac12 s_k'(t)\Delta^2+O(\Delta^3)$. Set $X:=s_k(t)\Delta+\tfrac12 s_k'(t)\Delta^2$; then $\mathbb E[X]=s_k\mathbb E[\Delta]+\tfrac12 s_k'\mathbb E[\Delta^2]$ and $\mathbb E[X^2]=s_k^2\mathbb E[\Delta^2]+O(\mathbb E[\Delta^3])$. Expanding $\log\mathbb E[e^X]=\mathbb E[X]+\tfrac12\big(\mathbb E[X^2]-\mathbb E[X]^2\big)+O(\Delta^3\text{-moments})$ (the cumulant expansion) gives
$$\bar s_k(t)=s_k\mathbb E[\Delta]+\tfrac12 s_k'\mathbb E[\Delta^2]+\tfrac12 s_k^2\big(\mathbb E[\Delta^2]-\mathbb E[\Delta]^2\big)+O(\Delta^3\text{-moments}),$$
which is (8). Taking real and imaginary parts with $s_k^2=(\rho_k^2-\omega_k^2)-2i\rho_k\omega_k$ yields the displayed component form; $s_k'=-\rho_k'+i\omega_k'$ contributes $-\tfrac12\rho_k'\mathbb E[\Delta^2]$ and $+\tfrac12\omega_k'\mathbb E[\Delta^2]$ respectively. Setting $s_k'\equiv0$ recovers the constant-pole map term for term.

*(e)* Condition throughout on $\mathcal F^{\mathrm{evt}}_j$; every inequality used in (b),(c) holds conditionally. $\qquad\blacksquare$

*Remark (identifiability and architecture).* Equation (8) exhibits a **three-way entanglement** in the event domain: an observer with access only to event-indexed data cannot separate the intrinsic pole $s_k(t)$, its drift $s_k'(t)$, and the gap moments $(\mathbb E[\Delta],\mathbb E[\Delta^2])$ from $\bar s_k$ alone. Disentanglement requires side information about the gap law — at minimum its first two moments — which is exactly what the gap-aware temporal tokens encode. This extends LLapDiff's renewal-based architectural argument to the time-varying setting, and sharpens it: for constant poles only $(\mathbb E[\Delta],\operatorname{Var}(\Delta))$ mattered; for chirped poles the *raw* second moment enters through the drift term, so gap mean and variance are needed **separately**, not merely in the combination $\operatorname{Var}(\Delta)$. It also yields a falsifiable prediction used in our stress tests: as the sampling-gap regime is varied at test time, the learned continuous-time pole *functions* $(\hat\rho_k(\cdot),\hat\omega_k(\cdot))$ should remain approximately invariant while the implied event-domain multipliers shift according to (8).*

---

## Notes for the write-up

- **Headline ablation.** "Chirp-modal **without** the residual MLP vs. LLapDiff (fixed pole **with** MLP)": show comparable or better CRPS/MSE while *deleting* the uncertified correction — the cleanest empirical rebuttal of the LLapDiff design, paired with Theorem B.
- **Strict-generalization claim.** State up front that fixed-pole/deterministic-mean LLapDiff is the constant-pole, $q_k\!\to\!0$ special case of (3),(7).
- **Expressivity (optional Thm).** Time-warped modal bases are dense in finite-energy chirp signals (relate to nonstationary Prony / adaptive Fourier decomposition) — answers "more expressive or just more parameters?". Be precise about what time variation buys: frequency/decay drift yes; per-mode amplitude growth **no** unless the Theorem B′ budget is enabled ($c_g>0$).
- **Theorem D is now a contribution, not future work.** It (i) strengthens LLapDiff's mean-stability under renewal averaging to *pathwise* contraction, (ii) derives the drift-corrected effective-pole map (8) with the new $\tfrac12 s_k'\mathbb E[\Delta^2]$ term, and (iii) turns the gap-aware-conditioning story into a falsifiable invariance prediction on learned pole trajectories. Pair it with a gap-regime stress test evaluated on the poles, not just CRPS.
- **Bounded-growth ablation.** Sweep $c_g\in\{0,\log2,\log5\}$; report where growth is used (learned $\gamma_k$ excursions) and whether trending/ramping datasets benefit. $c_g$ also preempts the "hard mean-reversion bias" objection.
- **Anticipated objection (mode coupling).** LLapDiff's modes are already block-diagonal, so we lose nothing relative to it; cross-mode interaction is carried by the nonlinear decoder (Koopman view). The principled coupled extension is the **Magnus expansion** (commutator corrections) — put in an appendix to signal you know the correct general-LTV tool.
- **Anticipated objection (diagonal time-varying SSMs).** The commuting-collapse mechanism also underlies diagonal SSMs and their input-dependent variants (S4D/S5/Mamba-style discretizations) and closed-form continuous-time networks; position CMD as (i) continuous-time closed form at *arbitrary irregular query times* (not a recurrence), (ii) inside a diffusion denoiser, (iii) with an explicit stability certificate on the model output, and (iv) analytic UQ — and cite that literature preemptively.
- **Implementation notes (hard-won; reproduction traps).** (i) The squared
  coefficients $(a^{\rho}_{km})^2$ of (P-exact) must **not** be zero-initialized:
  $a=0$ is a stationary point of the squaring ($d(a^2)/dW=2a\,h=0$), so a zero-init
  head receives exactly zero gradient and the "chirp" silently trains as constant
  poles — use an $\varepsilon$-init (e.g. std $10^{-4}$: $\sim10^{-8}$ from the LTI
  special case at init) and ship a gradient regression test. (ii) The basis
  frequencies are *cycles across the window*: scale them by the horizon length $L$,
  or at native horizons the oscillatory part of $\Phi_m$ is $O(1/2\pi f_m)$ against
  a linear term $O(L)$ and the chirp collapses to a constant-slope ramp. (iii) For
  the Gaussian NLL of §4.6, warm-start the mean from an MSE-trained checkpoint:
  from scratch, the small initial variance makes the mean gradient
  $(\hat z_0-z_0)/\sigma^2$ explode before the variance adapts. (iv) The likelihood
  uses the diagonal readout of (7), $\operatorname{Var}(z_{0,d})=\sum_k s_k(\tilde t)
  (c_{kd}^2+b_{kd}^2)$ — valid since $c_kc_k^\top+b_kb_k^\top=\|\xi^0_k\|^2\,
  C_kC_k^\top$, with the scale absorbed into $p_k^0,q_k$.
- **Honest limitations.** Local linearization is widened from a point (LLapDiff) to a warped trajectory, not eliminated; closed-form UQ assumes the linear-latent regime and covers aleatoric spread conditional on predicted parameters (no epistemic component); (P-exact) integration needs a basis with known antiderivatives (the *variance* integral $v_k$ additionally needs the 1-D quadrature of §4.5 even under (P-exact)); the stability certificate applies on $[0,H]$ from the earliest query (see the anchoring remark in §4.4).
