using OrdinaryDiffEq, LinearAlgebra


##### Spine model ODE connstants
const Δt = 10.0
const t_pre = 500.0
const t_post = t_pre + Δt
const τ₁ = 50.0
const τ₂ = 5.0
const τ_b = 20.0
const τ_k = 56.0
const τ_N = 100.0
const τ_Ca = 50.0
const τ_IP₃ = 0.140;
const vₑ = 1.0
const v_b = 60.0
const v_k = 2.0
const vᵣ = - 65.0
const Vᵣₑᵥ = 130.0
const α = 5.0
const K_d = 0.32
const P₀ = 0.8
const ḡ = 0.00103
const r_IP₃ = 0.2
const IP₃_b = 0.16
const t_inicio_IP₃ = 500
const IP₃_threshold = 50

# Spine model functions
Θ(x) = x > zero(x) ? one(x) : zero(x)

V(Ca, t) = vᵣ + BPAP(t) + EPSP_AMPA(t) + EPSP_SK(Ca, t)

EPSP_AMPA(t) = vₑ * Θ(t - t_pre) * (exp(-(t - t_pre)/τ₁) + exp(-(t - t_pre)/τ₁))
BPAP(t) = v_b * Θ(t - t_post) * exp(-(t - t_post)/τ_b)
EPSP_SK(Ca, t) = v_sk(t) * (Ca^α / (Ca^α + K_d^α))
v_sk(t) = v_k * Θ(t - t_post) * exp(-(t - t_post)/τ_k) + 0.018 # este 0.018 estaba en código de matlab pero no en la tesis

Inmda(V, t) = ḡ * f(t) * ℋ(V)
f(t) = P₀ * Θ(t - t_pre) * exp(-(t - t_pre)/τ_N) # en la tesis tiene dos términos con exponenciales (una rápida y otra lenta) pero en el codigo de matlab solo tiene una
ℋ(V) = (V - Vᵣₑᵥ) / (1 + (exp(-0.062 * V)/3.57))

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
const α₁ = 1.0.*(X.>=20)

# production of IP₃
const r_ip3 = zeros(N)
r_ip3[1] = IP₃_b

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

  mul!(MCa, M, Ca)
  mul!(MIP₃, M, IP₃)
  @.  DCa = D_Ca*MCa
  @. DIP₃ = D_IP₃*MIP₃

  @.  dCa = DCa + α₁ - Ca^2 * v_p/(k_p + Ca^2) + Ca_b^2 * v_p/(k_p + Ca_b^2)
  @. dIP₃ = DIP₃

  # spine model
   Ca_esp = Ca[1]
  IP₃_esp = IP₃[1]

        v = V(Ca_esp, t)
   dCa[1] =  -Inmda(v, t) - (1/τ_Ca) *  Ca_esp
  dIP₃[1] = Θ(t - t_inicio_IP₃) * ((1/τ_IP₃) * (IP₃_b - IP₃_esp) + Θ(v - IP₃_threshold) * r_IP₃)

  # source(ubicada en i) = 8 Pi D dx calcio/3
  #
  # d[Ca]/dt = D laplaciano[Ca] + sum_i source(ubicada en i)/(At dx) - [Ca]^2 v_p/(k_p + [Ca]^2)+ [Ca]_b^2 v_p/(k_p + [Ca]_b^2)
end

# Solve the ODE
tspan = (0.0, t_max)
t = 0.0:dt:t_max
prob = ODEProblem(g, u0, tspan, saveat = t)
sol = solve(prob, ROCK2())


using Plots
plot(sol, vars = 2)
plot(sol[:][1,2])


using Flux
sol_stacked = Flux.stack(sol[:], 1)

ip3 = sol_stacked[:,:,2]
plot(ip3[:,1])
cal = sol_stacked[:,:,1]
heatmap(cal)

plot(cal[:,1])

# using ImageView
