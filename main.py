
# Add the parent directory to the path so we can import from qoptimizer
from turtle import color
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
from joblib import Parallel, delayed
from tqdm import tqdm
import time

if __name__ == "__main__":
    dir_ = Path(__file__).resolve().parent




    n_spin = 7
    proposal = "qemcmc"
    repetitions = 20
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
    temps = np.logspace(np.log10(high_temp), np.log10(low_temp) , n_hops)

    model = get_models(n_spin, models_path=dir_/'models')[0]
    
    model_lowest = model.lowest_energy


    def run_single_repetition(model, proposal, quantum_args_dict, n_hops, temps):
        """Worker function to handle a single simulation run."""
        start = time.time()
        sa = QeSA(model, proposal=proposal, quantum_args_dict=quantum_args_dict)
        chain = sa.run_simulated_annealing(n_hops=n_hops, temp_scaling=temps)
        
        final_energy = chain.current_state.energy
        first_accepted_energy = chain.get_accepted_energies()[0]
        print("Final energy:", final_energy, flush=True)
        end = time.time()
        print("Time taken for this run: {:.2f} seconds".format(end - start), flush=True)
        min_energy = np.min(chain.get_accepted_energies()[0])
        return final_energy, first_accepted_energy, min_energy


    # n_jobs=-1 uses all available CPU cores
    # q_results = Parallel(n_jobs=-1)(
    #     delayed(run_single_repetition)(model, proposal, quantum_args_dict, n_hops, temps) 
    #     for _ in tqdm(range(repetitions))
    # )
    q_results = []
    for _ in tqdm(range(repetitions)):
        q_results.append(run_single_repetition(model, proposal, quantum_args_dict, n_hops, temps))


    # Unpack the results back into your lists
    q_optimal_energies_found, q_all_energies_found, q_min_energies_found = zip(*q_results)
    # Convert from tuples back to lists if necessary
    q_optimal_energies_found = list(q_optimal_energies_found)
    q_all_energies_found = list(q_all_energies_found)
    q_min_energies_found = list(q_min_energies_found)
    q_color = 'blue'





    # c_results = Parallel(n_jobs=-1)(
    #     delayed(run_single_repetition)(model, "local", None, n_hops, temps) 
    #     for _ in tqdm(range(repetitions*10))
    # )

    c_results = []
    for _ in tqdm(range(repetitions+10)):
        c_results.append(run_single_repetition(model, "local", None, n_hops, temps))

    # Unpack the results back into your lists
    c_optimal_energies_found, c_all_energies_found, c_min_energies_found = zip(*c_results)
    # Convert from tuples back to lists if necessary
    c_optimal_energies_found = list(c_optimal_energies_found)
    c_all_energies_found = list(c_all_energies_found)
    c_min_energies_found = list(c_min_energies_found)
    c_color = 'green'


    plt.plot(np.arange(0,n_hops+1),np.mean(np.array(c_all_energies_found), axis=0), color=c_color)
    for i in range(len(c_all_energies_found)):
        plt.plot(np.arange(0,n_hops+1), c_all_energies_found[i], alpha=0.03, color=c_color)
    plt.plot(np.arange(0,n_hops+1),np.mean(np.array(q_all_energies_found), axis=0), color=q_color)
    for i in range(len(q_all_energies_found)):
        plt.plot(np.arange(0,n_hops+1), q_all_energies_found[i], alpha=0.03, color=q_color)
    plt.xlabel('Hop Number')
    plt.ylabel('Energy')
    plt.title('Average Energy vs Hop Number | proposal: {}'.format(proposal))
    plt.axhline(y=model_lowest, color='r', linestyle='--', label='Exact optimal')
    plt.legend()
    plt.show()



    print("percentage of quantum runs that found the ground state:", np.sum(np.isclose(q_min_energies_found, model_lowest)/repetitions*100))
    print("percentage of classical runs that found the ground state:", np.sum(np.isclose(q_min_energies_found, model_lowest)/(repetitions+10)*100))

    # plot bar graph of results
    plt.figure(figsize=(8, 6))
    plt.hist(c_optimal_energies_found, bins=20, alpha=0.7, color=c_color, label='Local Results')
    plt.hist(q_optimal_energies_found, bins=20, alpha=0.7, color=q_color, label='SA Results')
    plt.axvline(x=model_lowest, color='red', linestyle='--', label='Exact optimal')    
    plt.xlabel('Energy found')
    plt.ylabel('Counts')
    plt.title('Optimal Energies Found by SA | proposal: {}'.format(proposal))
    plt.legend()
    plt.show()  

