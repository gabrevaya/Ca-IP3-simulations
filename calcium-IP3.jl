using OrdinaryDiffEq, LinearAlgebra


##### Spine model ODE connstants
const Δt = 10.0
const t_pre = 500.0
const t_post = t_pre + Δt
const τ₁ = 50.0
const τ₂ = 5.0
const τ_b = 20.0
const τ_k = 56.0
const τ_f = 200.0
const τ_s = 50.0
const τ_Ca = 50.0
const τ_IP₃ = 0.140;
const vₑ = 1.0
const v_b = 60.0
const v_k = 2.0
const vᵣ = - 65.0
const Vᵣₑᵥ = 130.0
const α = 5.0
const K_d = 0.32
const P₀ = 10.0
const ḡ = 0.00103
const r_IP₃ = 0.2
const IP₃_b = 0.16
const t_inicio_IP₃ = 500
const IP₃_threshold = 50
const I_f = 0.5
const I_s = 0.5
freq = 5
const t_est = collect(1:1000/freq:1000)

# Spine model functions
Θ(x) = x < zero(x) ? zero(x) : one(x)

V(Ca, t, t_pre, Δt) = vᵣ + BPAP(t, t_pre, Δt) + EPSP_AMPA(t, t_pre) + EPSP_SK(Ca, t, t_pre, Δt)

EPSP_AMPA(t, t_pre) = vₑ * Θ(t - t_pre) * (exp(-(t - t_pre)/τ₁) + exp(-(t - t_pre)/τ₂))
BPAP(t, t_pre, Δt) = v_b * Θ(t - (t_pre + Δt)) * exp(-(t - (t_pre + Δt))/τ_b)
EPSP_SK(Ca, t, t_pre, Δt) = v_sk(t, t_pre, Δt) * (Ca^α / (Ca^α + K_d^α))
v_sk(t, t_pre, Δt) = v_k * Θ(t - (t_pre + Δt)) * exp(-(t - (t_pre + Δt))/τ_k) + 0.018 # este 0.018 estaba en código de matlab pero no en la tesis

Inmda(Ca, t, t_pre, Δt) = ḡ * f(t, t_pre) * ℋ(V(Ca, t, t_pre, Δt))

# f(t, t_pre) = P₀ * Θ(t - t_pre) * exp(-(t - t_pre)/τ_N)
f(t, t_pre) = P₀ * Θ(t - t_pre) * (I_f * exp(-(t - t_pre)/τ_f) + I_s * exp(-(t - t_pre)/τ_s))
ℋ(V) = (V - Vᵣₑᵥ) / (1 + (exp(-0.062 * V)/3.57))

V(Ca, t, t_pre::Vector, Δt) = vᵣ + sum(BPAP.(t, t_pre, Δt) + EPSP_AMPA.(t, t_pre) + EPSP_SK.(Ca, t, t_pre, Δt))
f(t, t_pre::Vector) = sum(f.(t, t_pre))


# most efficient but less clear
EPSP_AMPA(t, t_pre) = t < t_pre ? zero(t) : vₑ * (exp(-(t - t_pre)/τ₁) + exp(-(t - t_pre)/τ₂))
BPAP(t, t_pre, Δt) = t < (t_pre + Δt) ? zero(t) : v_b * exp(-(t - (t_pre + Δt))/τ_b)
EPSP_SK(Ca, t, t_pre, Δt) = t < (t_pre + Δt) ? zero(t) : (v_k * exp(-(t - (t_pre + Δt))/τ_k) + 0.018) * (Ca^α / (Ca^α + K_d^α))
f(t, t_pre) = t < t_pre ? zero(t) : P₀ * (I_f * exp(-(t - t_pre)/τ_f) + I_s * exp(-(t - t_pre)/τ_s))


##### PDE constants ####
const D_Ca = 20.0
const D_IP₃ = 250.0
const v_p = 9e-4 # CHECK THIS VALUE
const k_p = 0.01 # CHECK THIS VALUE
const Ca_b = 0.042
const IP₃_b = 0.16 # chequear la concentración de IP₃ basal

t_max = 1000.0
# dt = 1e-5 # at least for Euler time integration, dt should be < 0.45*dx^2/D_ip3
dt = 1.0
x_max = 50
dx = 0.1
const X = collect(0:dx:x_max)
N = length(X)

# production of Ca as a function of space
const α₁ = 0.01.*(X.>=300)


# production of IP₃
const R_IP₃ = zeros(N)
R_IP₃[1] = 2*r_IP₃

const M = collect(Tridiagonal([1.0 for i in 1:N-1],[-2.0 for i in 1:N],[1.0 for i in 1:N-1]))
M[2,1] = 2.0
M[end-1,end] = 2.0


# Define the initial condition as normal arrays
# u0 = ones(N,2)*Ca_b # search for the initial condition of IP3
u0 = repeat([Ca_b IP₃_b], N)
# u0 = repeat([Ca_b 0], N)

const MCa = zeros(N);
const DCa = zeros(N);
const MIP₃ = zeros(N);
const DIP₃ = zeros(N);


# Define the discretized PDE as an ODE function
function g(du,u,p,t)
       Ca = @view  u[:,1]
      IP₃ = @view  u[:,2]
      dCa = @view du[:,1]
     dIP₃ = @view du[:,2]
       Δt = p

  mul!(MCa, M, Ca)
  mul!(MIP₃, M, IP₃)
  @.  DCa = D_Ca*MCa
  @. DIP₃ = D_IP₃*MIP₃

  @.  dCa = DCa + α₁ - Ca^2 * v_p/(k_p + Ca^2) + Ca_b^2 * v_p/(k_p + Ca_b^2)
  @. dIP₃ = DIP₃ + Θ(t - t_inicio_IP₃) * R_IP₃

  # spine model
   dCa[1] =  -Inmda(Ca[1], t, t_est, Δt) - (1/τ_Ca) * Ca[1]

  # source(ubicada en i) = 8 Pi D dx calcio/3
  #
  # d[Ca]/dt = D laplaciano[Ca] + sum_i source(ubicada en i)/(At dx) - [Ca]^2 v_p/(k_p + [Ca]^2)+ [Ca]_b^2 v_p/(k_p + [Ca]_b^2)
end

# Solve the ODE
tspan = (0.0, t_max)
t = 0.0:dt:t_max
Δt = 5.0
prob = ODEProblem(g, u0, tspan, Δt, saveat = t)
sol = solve(prob, ROCK2())

using BenchmarkTools
@btime solve($prob, $ROCK2())

using Plots
plot(sol, vars = 1)

using Flux
sol_stacked = Flux.stack(sol[:], 1)

ip3 = sol_stacked[:,:,2]
cal = sol_stacked[:,:,1]

plot(ip3[:,400])

heatmap(cal)
plot(cal[:,1])
