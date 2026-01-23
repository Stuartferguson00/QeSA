
# Add the parent directory to the path so we can import from qoptimizer
import os, sys, warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qesa.qesa import QeSA
from qesa.utils import get_models
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from tqdm import trange

if __name__ == "__main__":
    dir_ = Path(__file__).resolve().parent




    n_spin = 7
    proposal = "local"
    repetitions = 1000
    n_hops = 100
    
    gamma = (0.25,0.6)
    time_ = (2,20)
    delta_time = 0.8
    quantum_args_dict = {'gamma': (0.25, 0.6), 'time': (2, 20), 'delta_time': 0.8}
    

    if proposal == "qemcmc":
        gamma = (0.25,0.6)
        time_ = (2,20)
        delta_time = 0.8
        quantum_args_dict = {'gamma': (0.25, 0.6), 'time': (2, 20), 'delta_time': delta_time}
    else:
        quantum_args_dict = None



    high_temp = 10.0
    low_temp = 0.01
    temps = np.logspace(high_temp, low_temp, n_hops)

    model = get_models(n_spin, models_path=dir_/'models')[0]
    
    model_lowest = model.lowest_energy
    optimal_energies_found = []


    for rep in trange(repetitions):
        sa = QeSA(model, proposal = proposal, quantum_args_dict = quantum_args_dict)
        chain = sa.run_simulated_annealing(n_hops = n_hops, temp_scaling = temps)
        final_energy = chain.current_state.energy
        min_energy_found = final_energy
        optimal_energies_found.append(min_energy_found)
    print("percentage of runs that found the ground state:", np.sum(np.isclose(optimal_energies_found, model_lowest)/repetitions*100))
    # plot bar graph of results
    plt.figure(figsize=(8, 6))
    plt.hist(optimal_energies_found, bins=20, alpha=0.7, color='blue', label='SA Results')
    plt.axvline(x=model_lowest, color='red', linestyle='--', label='Exact optimal')    
    plt.xlabel('Energy found')
    plt.ylabel('Counts')
    plt.title('Optimal Energies Found by SA | proposal: {}'.format(proposal))
    plt.legend()
    plt.show()  

