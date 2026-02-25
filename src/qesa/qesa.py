"""
Quantum Enhanced Simulated Annealing (QeSA) Module

This module implements a hybrid quantum-classical simulated annealing algorithm
that combines quantum MCMC proposals with classical simulated annealing optimization.
The algorithm supports multiple proposal methods and temperature scheduling schemes.

Author: Stuart Ferguson
Date: 2025
Dependencies: qemcmc 0.3.31
"""
import qemcmc as qemcmc
qemcmc.__version__
from qemcmc.model.energy_model import EnergyModel
from qemcmc.sampler.qe_mcmc import QeMCMC
from qemcmc.sampler.classical_mcmc import ClassicalMCMC
from qemcmc.utils.helpers import get_random_state
from qemcmc.utils.helpers import MCMCState, MCMCChain
import typing
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional


class QeSA(QeMCMC, ClassicalMCMC):
    """
    Quantum Enhanced Simulated Annealing (QeSA) class.
    
    This class implements a hybrid optimization algorithm that combines quantum MCMC
    proposal methods with classical simulated annealing. It inherits from both QeMCMC
    and ClassicalMCMC to leverage quantum and classical proposal mechanisms.
    
        
    """
    
    def __init__(
        self,
        model: EnergyModel,
        proposal: str,
        quantum_args_dict: Optional[Dict[str, typing.Any]] = None
    ) -> None:
        """
        Initialize the QeSA optimizer.
        
        Args:
            model: The energy model/Hamiltonian to optimize. Must have a get_energy() method.
            proposal (str): The proposal method to use. Options:
                - 'local': Classical local moves (single spin flips)
                - 'uniform': Classical uniform random proposals  
                - 'qemcmc': Quantum enhanced MCMC proposals
            quantum_args_dict (Dict, optional): Dictionary containing quantum parameters.
                Required for 'qemcmc' proposal:
                - 'gamma': Quantum parameter controlling proposal distribution
                - 'time': Evolution time for quantum proposals
                - 'delta_time': Time step (optional, defaults to 0.8)
                
        Raises:
            ValueError: If proposal method is invalid or required quantum args are missing
            KeyError: If required quantum arguments are not provided for qemcmc
        """
        # Store the proposal method for reference
        self.proposal_method = proposal
        
        # Initialize based on proposal method
        if proposal == "local":
            # Classical local moves - initialize with classical MCMC
            ClassicalMCMC.__init__(self, model, np.nan, proposal)
        elif proposal == "uniform":
            # Classical uniform random proposals
            ClassicalMCMC.__init__(self, model, np.nan, proposal)
        elif proposal == "qemcmc":
            # Quantum enhanced proposals - need quantum parameters
            if quantum_args_dict is None:
                raise ValueError("quantum_args_dict is required for qemcmc proposal method")
            
            try:
                gamma = quantum_args_dict['gamma']
                time = quantum_args_dict['time']
                delta_time = quantum_args_dict.get('delta_time', 0.8)  # Default to 0.8 if not provided
            except KeyError as e:
                raise ValueError(f"Missing required quantum argument: {e}")
            
            # Initialize quantum MCMC parent class
            super().__init__(model, gamma, time, np.nan, delta_time)
            # Set the update function to use quantum proposals
            #self.update = self.get_s_prime
        else:
            raise ValueError(f"Invalid proposal method: {proposal}. Choose from 'local', 'uniform', 'qemcmc'")
        
    
    
    
    def run_simulated_annealing(self,
        n_hops: int,
        temp_scaling: np.ndarray,
        initial_state: str = "",
        name: str = "SA",
        verbose: bool = False,
        sample_frequency: int = 1) -> MCMCChain:
        """
        Run simulated annealing with predefined temperature schedule.
        
        This method implements simulated annealing where the temperature follows
        a predefined schedule specified by the temp_scaling array. The algorithm
        runs for exactly n_hops iterations.
        
        Args:
            n_hops (int): Number of MCMC steps to perform.
            temp_scaling (np.ndarray): Temperature schedule array of length n_hops.
                Each element specifies the temperature for that iteration.
            initial_state (str, optional): Initial state bitstring. If empty or wrong length,
                a random state is generated. Defaults to "".
            name (str, optional): Name for the MCMC chain. Defaults to "SA".
            verbose (bool, optional): Enable verbose output with progress bar. Defaults to False.
            sample_frequency (int, optional): Frequency of state sampling. Defaults to 1.
                
        Returns:
            MCMCChain: Chain containing sampled states during annealing process.
            
        Algorithm:
            1. Initialize state
            2. For each of n_hops iterations:
                a. Propose new state using update method
                b. Accept/reject using Metropolis criterion with scheduled temperature
                c. Sample state if at sampling frequency
            3. Add final state to chain
            4. Return collected samples
            
        """

        
        
        # Handle initial state - generate random if not provided or invalid
        if len(initial_state) == 0:
            initial_state = MCMCState(get_random_state(self.n_spins), accepted=True)
        elif len(initial_state) != self.n_spins:
            initial_state = MCMCState(get_random_state(self.n_spins), accepted=True)
        else:
            initial_state = MCMCState(initial_state, accepted=True)
        
        # Calculate initial energy and set up initial state
        energy_s = self.model.get_energy(initial_state.bitstring)
        initial_state.energy = energy_s
        initial_state.position = 0

        # Set current state to initial state
        current_state: MCMCState = initial_state

        # Print starting information if verbose
        if verbose:
            print("starting with: ", current_state.bitstring, "with energy:", energy_s)

        # Initialize MCMC chain to store samples
        mcmc_chain = MCMCChain([current_state], name=name)





        # ==================== MAIN ANNEALING LOOP ====================
        
        # Fixed iteration loop with progress bar
        for i in tqdm(range(0, n_hops), desc='running simulated annealing steps...', disable=not verbose):
            # Generate new proposed state
            s_prime = self.update(current_state.bitstring)

            # Calculate energy of proposed state
            energy_sprime = self.model.get_energy(s_prime)
            
            # Get temperature from predefined schedule
            temp = temp_scaling[i]
                        
            # Metropolis-Hastings acceptance test with scheduled temperature
            accepted = self.test_accept(energy_s, energy_sprime, temperature=temp)
                        
            # Update current state if proposal was accepted
            if accepted:
                current_state = MCMCState(s_prime, accepted, energy_sprime, position=i)
                energy_s = energy_sprime 
            
            # Sample state at specified frequency (skip initial state i=0)
            if i // sample_frequency == i / sample_frequency and i != 0:
                mcmc_chain.add_state(current_state)
        
        # Always add the final state to the chain
        mcmc_chain.add_state(current_state)
        
        # Return collected samples
        return mcmc_chain
    

