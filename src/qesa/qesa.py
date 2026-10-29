"""
Quantum Enhanced Simulated Annealing (QeSA) Module

This module implements a hybrid quantum-classical simulated annealing algorithm
that combines quantum MCMC proposals with classical simulated annealing optimization.
The algorithm supports multiple proposal methods and temperature scheduling schemes.

Author: Stuart Ferguson
Date: 2026
Dependencies: qemcmc 0.6.0
"""
import qemcmc as qemcmc
from qemcmc.model.energy_model import EnergyModel
from qemcmc.sampler import proposal
from qemcmc.sampler.runners import MCMCRunner
from qemcmc.utils.helpers import get_random_state
from qemcmc.utils.helpers import MCMCState, MCMCChain
import numpy as np
from tqdm import tqdm


class QeSA(MCMCRunner):
    """
    Quantum Enhanced Simulated Annealing (QeSA) class.
    
    This class implements a hybrid optimization algorithm that combines quantum MCMC
    proposal methods with classical simulated annealing. It inherits from both QeMCMC
    and ClassicalMCMC to leverage quantum and classical proposal mechanisms.
    
        
    """
    
    def __init__(
        self,
        model: EnergyModel,
        proposal: proposal,
    ) -> None:
        """
        Initialize the QeSA optimizer.
        
        Args:
            model: The energy model/Hamiltonian to optimize.
            proposal (proposal): The qemcmc.proposal object to use.

        """
        super().__init__(model, np.nan)
        self.proposal = proposal
        
        
        
    
    
    
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
            initial_state = MCMCState(get_random_state(self.model.n_spins), accepted=True)
        elif len(initial_state) != self.model.n_spins:
            initial_state = MCMCState(get_random_state(self.model.n_spins), accepted=True)
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
            # Get temperature from predefined schedule
            temp = temp_scaling[i]
                        
            s_prime = self.proposal.update(current_state.bitstring)
            energy_sprime = self.model.get_energy(s_prime)
            accepted = self.is_accepted(energy_s, energy_sprime, temperature=temp)

            if accepted:
                energy_s = energy_sprime
                current_state = MCMCState(s_prime, accepted, energy_s, position=i)

            if i % sample_frequency == 0 and i != 0:
                mcmc_chain.add_state(MCMCState(current_state.bitstring, True, energy_s, position=i))

        
        # Always add the final state to the chain
        mcmc_chain.add_state(current_state)
        
        # Return collected samples
        return mcmc_chain
    

