using OrdinaryDiffEq

# Ecuacion diferencial para la concentracion de Calcio en la espina
# postsinaptica

freq = 5 #Hz
const Δt = 10.0
const t_pre = 500.0
const t_post = t_pre + Δt
const τ₁ = 50.0
const τ₂ = 5.0
const τ_b = 20.0
const τ_k = 56.0
const τ_N = 100.0
const τ_Ca = 50.0
const vₑ = 1.0
const v_b = 60.0
const v_k = 2.0
const vᵣ = - 65.0
const Vᵣₑᵥ = 130.0
const α = 5.0
const K_d = 0.32
const P₀ = 10.0
const ḡ = 0.00103
const t_est = collect(1:1000/freq:1000)
const τ_f = 200.0
const τ_s = 50.0
const I_f = 0.5
const I_s = 0.5

Θ(x) = x < zero(x) ? zero(x) : one(x)

V(Ca, t, t_pre, Δt) = vᵣ + BPAP(t, t_pre, Δt) + EPSP_AMPA(t, t_pre) + EPSP_SK(Ca, t, t_pre, Δt)

EPSP_AMPA(t, t_pre) = vₑ * Θ(t - t_pre) * (exp(-(t - t_pre)/τ₁) + exp(-(t - t_pre)/τ₂))
BPAP(t, t_pre, Δt) = v_b * Θ(t - (t_pre + Δt)) * exp(-(t - (t_pre + Δt))/τ_b)
EPSP_SK(Ca, t, t_pre, Δt) = v_sk(t, t_pre, Δt) * (Ca^α / (Ca^α + K_d^α))
v_sk(t, t_pre, Δt) = v_k * Θ(t - (t_pre + Δt)) * exp(-(t - (t_pre + Δt))/τ_k) + 0.018 # este 0.018 estaba en código de matlab pero no en la tesis

Inmda(Ca, t, t_pre, Δt) = ḡ * f(t, t_pre) * ℋ(V(Ca, t, t_pre, Δt))

# f(t, t_pre) = P₀ * Θ(t - t_pre) * exp(-(t - t_pre)/τ_N)
f(t, t_pre) = P₀ * (I_f * Θ(t - t_pre) * exp(-(t - t_pre)/τ_f) + I_s * Θ(t - t_pre) * exp(-(t - t_pre)/τ_s))
ℋ(V) = (V - Vᵣₑᵥ) / (1 + (exp(-0.062 * V)/3.57))

V(Ca, t, t_pre::Vector, Δt) = vᵣ + sum(BPAP.(t, t_pre, Δt) + EPSP_AMPA.(t, t_pre) + EPSP_SK.(Ca, t, t_pre, Δt))
f(t, t_pre::Vector) = sum(f.(t, t_pre))


dCa(Ca, Δt, t) = - Inmda(Ca, t, t_est, Δt) - (1/τ_Ca) * Ca

Ca₀ = 0.0
t = 0.0:1:1000.0
Δt = 10.0
prob = ODEProblem(dCa, Ca₀, (0.0,1000.0), Δt, saveat = t)
sol1 = solve(prob,Tsit5())

using Plots
plot(sol1 , vars = 1, label = "Δt = $Δt ms", title = "5 Hz")

Δt = -20.0
prob = remake(prob, p = Δt)
sol2 = solve(prob,Tsit5())
plot!(sol2 , vars = 1, label = "Δt = $Δt ms")

savefig("5Hz.pdf")
