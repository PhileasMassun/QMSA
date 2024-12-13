using  TensorKit, RandomMatrix, Plots, LinearAlgebra;

function apply_H(ρ,Jm,Je)
    #extract |ρ> from the padded state
    #ρ = ρ[:, 1, 1, 1, 1]
    
    energy = 0
    n = length(ρ)
    N = n/2
    for i in 1:n
        if i <= N
            energy += -1* Jm * ρ[i]
        else
            energy += -1* Je * ρ[i]
        end
    end
    return energy
end

function create_initial(m)
    #creates a 2^m sized lattice vector, which is consistent with a lattice of size 2^(m-1)
    state = zeros(2^m)
    for i in 1:2^m
        t = abs(randn())
        if t>0.5
            state[i] = 1
        else
            state[i] = -1
        end
    end
    return state 
end

function Draw_unitary(m)
    #naive implementation, can be improved by not selecting same site twice etc.
    #satisfies the prob measure requirement i think
    # Initialize a 2N x 2N identity matrix
    #TODO resize to 2^m
    unitary = Matrix{ComplexF64}(I(2^m))
    #unitary = Matrix(unitary)  # Convert to a mutable array

    # Draw a random integer between 1 and 2N for the number of eigenvalue flips
    n = rand(1:2^m)


    # For each flip, pick a random site and flip the sign
    for i in 1:n
        site = rand(1:2^m)  
        unitary[site, site] = -1
    end

    return unitary
end

function get_degeneracy(E,m,J)
    #let us for now assume Je=Jm=j
    num_flips::Int = E/(2*J)
    #println("the number of flips: ",num_flips)
    degen = factorial(big(2^m))/(factorial(big(2^m-num_flips))*factorial(big(num_flips)))
    return degen
end

function gibbs_energy_distr(E,m,J,β)
    g = get_degeneracy(E,m,J)
    return (g*exp(-β*E))
end
function gibbs_norm(m,J,β)
    Z = 0
    for E in 0:2:2*2^m
        #println("current energy: ",E)
        Z += gibbs_energy_distr(E,m,J,β)
        #println("current Z: ", Z)
    end
    return Z 
end

function get_frequencies(energies,m)
    frequencies : zeros(2^m)
    for i in 0:2:2*2^m
        amount = count(j->(j==i),energies)
        frequencies[i] = amount
    end
    return frequencies
end
function get_index(state,update)
    ind_state = update * state
    l = length(ind_state)
    for entry in 1:l 
        if entry == 1
            return entry,false 
        end
    end
    return 0,true
end

function normalise(state,index,GS)
    if GS
        return real.(-1* state/state[index])
    end
    return real.(state/state[index])
end

function apply_Q(state,C,Φ,W, m,r)
    @tensor state_to_be_measured[-1,-2,-3,-4,-5] := W[-2,-3,-4;d,e,f]*Φ[-1,d;b,c] * C[b,a] * state[a, c, e, f, -5]
    measured_state, result = measure_acceptance_qubit(state_to_be_measured)
    Φ_dag = reshape(reshape(Φ,(2^(m+r),2^(m+r)))',(2^m,2^r,2^m,2^r))
    W_dag = reshape(reshape(W,(2^(2r+1),2^(2r+1)))',(2^r,2^r,2,2^r,2^r,2))
    C_dag = C'

    @tensor new_state[-1,-2,-3,-4,-5] := C_dag[-1,f]*Φ_dag[f,-2;d,e]*W_dag[e,-3,-4;a,b,c]*measured_state[d,a,b,c,-5]
    return new_state, result
end

function apply_P(state,Φ,V, m,r)
    @tensor state_to_be_measured[-1,-2,-3,-4,-5] := V[-2,-3,-5;d,e,f]*Φ[-1,d;a,c] * state[a, c, e, -4, f]
    measured_state, result = measure_rewind_qubit(state_to_be_measured)
    Φ_dag = reshape(reshape(Φ,(2^(m+r),2^(m+r)))',(2^m,2^r,2^m,2^r))
    V_dag = reshape(reshape(V,(2^(2r+1),2^(2r+1)))',(2^r,2^r,2,2^r,2^r,2))

    @tensor new_state[-1,-2,-3,-4,-5] := Φ_dag[-1,-2;d,a]*measured_state[d,a,-3,-4,-5]
    return new_state, result
end

function apply_L(state,C,Φ,W, m,r)
    Φ_dag = reshape(reshape(Φ,(2^(m+r),2^(m+r)))',(2^m,2^r,2^m,2^r))

    @tensor new_state[-1,-2,-3,-4,-5] := Φ_dag[-1,-2;g,h]*W[h,-3,-4;d,e,f]*Φ[g,d;b,c] * C[b,a] * state[a, c, e, f, -5]
    return new_state
end

function prepare_initial_energy_register(state, Φ)
    @tensor new_state[-1, -2, -3, -4, -5] := Φ[-1, -3; a, b] * state[a, -2, b, -4, -5] 
    return new_state
end

function get_W(r, β, t)
    #constructs a 2^rx2^r matrix where every entry at pos (i,j) is the 2x2 matrix as defined in (2).
    #this essentialy creates all such possible matrices for all possible pairs of E differences inside the range 2^r
    gibbs_energy_difference(z_2, z_1) = min(1, ℯ^(-β * (2π) * (z_1 - z_2) / (2^r * t))) # Why the extra factor of 2??

    W_mat = zeros(ComplexF64, 2^r, 2^r, 2, 2^r, 2^r, 2)
    for i in 1:2^r
        for j in 1:2^r
            W_mat[i, j, :, i, j, :] = [[(1 - gibbs_energy_difference(i, j))^0.5, gibbs_energy_difference(i, j)^0.5]; [gibbs_energy_difference(i, j)^0.5, -(1 - gibbs_energy_difference(i, j))^0.5]]
        end
    end

    return W_mat
end

function get_V(r, μ)
    #function to indicate "closeness" of energy states defined by threshhold μ
    indicator_function(z_2, z_1) = abs(z_1 - z_2) < μ ? 1 : 0

    V_mat = zeros(ComplexF64, 2^r, 2^r, 2, 2^r, 2^r, 2)
    for i in 1:2^r
        for j in 1:2^r
           #generate a 2x2 block matrix for each index i,j, the form of which is dependant on the indicator.
           #if the indicator is 1 the 2x2 is ((0,1),(1,0)), if indicator is 0 it is de 2x2 identity
            V_mat[i, j, :, i, j, :] = [[1 - indicator_function(i, j), indicator_function(i, j)]; [indicator_function(i, j), 1 - indicator_function(i, j)]]
        end
    end

    return V_mat
end

function get_iQFT(r)
    iQFT_mat = zeros(ComplexF64, 2^r, 2^r)
    for i in 1:2^r
        for j in 1:2^r
            iQFT_mat[i, j] = ℯ^(1im * (i - 1) * (j - 1) * 2π / 2^r) / 2^(r / 2)
        end
    end
    return iQFT_mat
end

function get_U_exp(state, r, m,J)
    #all U has to do is extract alpha. i can find alpha with a function call to the state, this is fine as its an eigenvect so no
    #collapse. then u exp mat can stay the same to construct the QPE matrix.
    #U_mat = exp(1im * H_mat)
    Jm = J
    Je=J
    #extract the state vector for the H
    ρ_state = state[:, 1, 1, 1, 1]
    E = apply_H(ρ_state,Jm,Je) +2^m*J #account for redefinition of E-scale
    U_vec = exp(1im*E) * ones(2^m)
    U_mat = diagm(U_vec)
    U_exp_mat = zeros(ComplexF64, 2^m, 2^r, 2^m, 2^r)
    for i in 1:2^r
        U_exp_mat[:, i, :, i] = U_mat^(i - 1)
    end
    return U_exp_mat
end

function get_Hadamard_gates(r)
    H = 1 / √2 * [[1 1]; [1 -1]]
    H_gate = TensorMap(H, ℂ^2 → ℂ^2)
    H_gates = H_gate
    for i in 1:r-1
        H_gates = H_gates ⊗ H_gate
    end
    H_gates_mat = Matrix(reshape(H_gates[], 2^r, 2^r))
    return H_gates_mat
end

function get_Φ(state, r, m,J)
    #QPE
    U_exp = get_U_exp(state, r, m,J)
    iQFT = get_iQFT(r)
    hadamard_gates = get_Hadamard_gates(r)
    #check if this contraction still makes sense when H_mat is replaced by function
    @tensor Φ[-1, -2; -3, -4] := iQFT[-2, a] * U_exp[-1, a, -3, b] * hadamard_gates[b, -4]
    return Φ
end

function measure_acceptance_qubit(state)
    result = 0
    random_number = rand()
    prob = norm(state[:,:,:,1,:])^2
    if random_number< prob
        state[:,:,:,2,:] .= 0
        state = state/sqrt(prob)
        result = 0
    else
        state[:,:,:,1,:] .= 0
        state = state/sqrt(1-prob)
        result =1
    end
    # state = state/norm(state)

    return state, result
end

function measure_rewind_qubit(state)
    result = 0
    random_number = rand()
    print(norm(state[:,:,:,:,1]))
    prob = norm(state[:,:,:,:,1])^2
    println(prob)
    if random_number< prob
        state[:,:,:,:,2] .= 0
        state = state/sqrt(prob)
        result = 0
    else
        state[:,:,:,:,1] .= 0
        state = state/sqrt(1-prob)
        result =1
    end

    return state, result
end

function apply_metropolis_step(ρ_state, Φ,W,V,m,r,n,J)
    state =  add_ancillas(ρ_state,r)
    #phi no longer statically defined through H_mat but through lattice energy directly, need to create new one each step.
    Φ_initial = get_Φ(state,r,m,J)
    state = prepare_initial_energy_register(state, Φ_initial) #prepare reg 2 with pre update E value
    C = Draw_unitary(m)
    @tensor prop_update[-1] := C[-1,a] * ρ_state[a]
    padded_update = add_ancillas(prop_update,r)
    Φ_new = get_Φ(padded_update,r,m,J)
    index,GS = get_index(ρ_state,C)
    #check below if at any step new phi is needed or old phi is needed
    for i in 1:n+1
        print("Q")
        state, Q_result = apply_Q(state,C,Φ_new,W, m,r)
        
        if Q_result == 1 && i ==1
            print("L")
            state = apply_L(state,C,Φ_new,W, m,r)
            return measure_all_ancillas(state,r,m), true, false,index,GS
        end
        print("P")
        state, P_result = apply_P(state,Φ_initial,V, m,r)
        if P_result == 1
            return measure_all_ancillas(state,r,m), false, false,index,GS
        end
    end
    println("Rewinding failed")
    return measure_all_ancillas(state,r,m), false, true,index,GS
end

function measure_all_ancillas(state, r,m)
    state = reshape(state, (2^m,2^r,2^r,2,2))

    # Measure Q and P ancilla qubits
    random_number = rand()
    prob = norm(state[:,:,:,:,1])^2
    if random_number< prob
        state = state[:,:,:,:,1]/norm(state[:,:,:,:,1])
    else
        state = state[:,:,:,:,2]/norm(state[:,:,:,:,2])
    end

    random_number = rand()
    prob = norm(state[:,:,:,1])^2
    if random_number< prob
        state = state[:,:,:,1]/norm(state[:,:,:,1])
    else
        state = state[:,:,:,2]/norm(state[:,:,:,2])
    end


    # Measuring QPE_ancilla_registers
    random_number = rand()
    cum_prob = 0
    for i in 1:2^r
        cum_prob += norm(state[:,:,i])^2
        if random_number< cum_prob
            state = state[:,:,i]/norm(state[:,:,i])
            break
        end
    end
    random_number = rand()
    cum_prob = 0
    for i in 1:2^r
        cum_prob += norm(state[:,i])^2
        if random_number< cum_prob
            state = state[:,i]/norm(state[:,i])
            break
        end
    end
    return state
end

function add_ancillas(ρ_state,r)
    """
    create the combined state matrix consisting of 5 registers:
    - |ρ>  : a 2^m sized vector of the actual physical state.
    - |E_i>: a 2^r sized register encoding energy of current state
    - |E_k>: a 2^r sized register encoding energy of new proposed state.
    - |q>  : a 1 qubit sized register encoding the acceptance ancilla.
    - |s>  : a 1 qubit sized register encoding the Q/P ancilla, denoting the measurement result.
    """
    empty_state = zeros(ComplexF64,(2^r,2^r,2,2))
    empty_state[1] = 1
    @tensor state[1,2,3,4,5]:= ρ_state[1] * empty_state[2,3,4,5] # m+2r+2 qubits
    return state
end

function get_initial_state(m)
    ρ_state = zeros(ComplexF64,2^m)
    ρ_state[1] = 1
    return ρ_state
end

function run_metropolis(;r, n, k, Lat_size, β,J)
    E = LinRange(0,2*Lat_size,Lat_size+1)         #energy range with GS redefined to have 0 energy
    m::Int = log(2,Lat_size)+1
    println("m: ",m)
    Z = gibbs_norm(m,J,β)                #normalisation factor
    p = gibbs_energy_distr.(E,m,J,β)                          # gibbs weights for E
    μ::Float64 = 1
    t::Float64 = 1
    ρ_state= create_initial(m)
    Φ = get_Φ(ρ_state, r, m,J) # unitary on r+m qubits
    W = get_W(r, β, t) # unitary on 2r+1 qubits
    V = get_V(r, μ) # unitary on 2r+1 qubits
    step = 0
    num_of_failures = 0
    density_matrices = zeros(ComplexF64, (k, 2^m, 2^m))
    steps_accepted = zeros(k)
    Energies = zeros(k)
    fidelity = zeros(k)

    while step < k
        println("Step: ", step, " ρ_state: ", ρ_state)
        ρ_state, accepted, failed,index,GS = apply_metropolis_step(ρ_state, Φ,W,V, m,r,n,J)
        println("Step: ", step, " ρ_state: ", ρ_state)
        if failed
            step = 0
            ρ_state = create_initial(m) #might have to change this, but seems fine.
            num_of_failures += 1
        else
            #here,also store energy value to compare with exp(-βE) function instead of direct gibbs state, redefine fidelity in that way
            density_matrices[step+1,:,:] = ρ_state*ρ_state'
            #calculates overlap of state with Gibbs? i can replace with straight E diff? yes
            #define a "measure density matrix" where we remove the complex-phase to calc lattice energy
            ρ_state = normalise(ρ_state,index,GS)
            Energies[step+1] = apply_H(ρ_state,J,J)
            #overlapswithH = [eigvecsH[:, i]'* ρ_state for i in 1:length(eigvalsH)]
            #print(abs.(overlapswithH))
            #fidelity[step+1] = abs(overlapswithH'* exp.(-β*eigvalsH))
            steps_accepted[step+1] = accepted
            step += 1
        end
    end
    
    return density_matrices, steps_accepted, num_of_failures, Energies
end

function get_target(β, H_mat)
    target_state = exp(-β * H_mat) / tr(exp(-β * H_mat))
    # println(target_state)
    return target_state
end

function get_error_norms(r,n)
    repeats = 10
    Lat_size = 12
    β::Float64 = 
    J=1
    densities, num_of_failures = run_metropolis(r=r,n=n,k=20, Lat_size = Lat_size, β=β,J=J)
    for _ in 1:repeats
        density_matrix, temp_num_of_failures = run_metropolis(r=r,n=n,k=20, Lat_size = Lat_size, β=β,J=J)
        num_of_failures += temp_num_of_failures
        densities += density_matrix
    end
    densities = densities/(repeats+1)
    print(num_of_failures)
    success_prob = 1 / (num_of_failures/(repeats+1) + 1)
    error_norms = zeros(Float64, 20)
    for i in 1:20
        error_norms[i] = norm(densities[i,:,:] / tr(densities[i,:,:]) - target_state,2) #TODO redefine error norm, as difference between gibbs distr and obtained distr?
    end
    return error_norms,success_prob
end

function plot_graph()
    r_max=4
    n_max =2
    y = zeros(Float64, (20,n_max,r_max))
    success_probs = zeros(Float64, ( n_max, r_max))
    label = []
    for r in 1:r_max
        for n in 1:n_max
            @time y[:,n, r], success_probs[n,r] = get_error_norms(r,n)
            push!(label, "n*=$n r=$r")
        end
    end
    label = permutedims(label)
    print(label)
    # y = reshape(y, 20,r_max*n_max)
    display(plot(1:20, y,ylims=(0,1), label=label,color_palette=:okabe_ito))
    print(success_probs)
    # plot(1:r_max, fp,  label=label)
    display(heatmap(success_probs, ylabel="n*", xlabel="r", clims=(0,1), title="Success Prob"))
    display(heatmap(y[20,:,:], ylabel="n*", xlabel="r", title="Error",clims=(0,1)))
end


Lat_size = 16
J=1
Es = LinRange(0,2*Lat_size,Lat_size+1)
β::Float64 = 1
@time density_matrices, steps_accepted, num_of_failures,Energies =  run_metropolis(r=4,n=3,k=20, Lat_size = Lat_size, β=β,J=J)
plot(steps_accepted,Energies)
