import os
import random
import numpy as np
import math

class AnnealSimulation():
    def __init__(self, objective, space, initial_state, initial_temperature, max_iter = 1000, plato_criterion = 50, log_directory = '', task_name = ''):
        self.space = space
        self.initial_state = initial_state
        self.ranges = [len(x) for x in space.values()]
        self.dim_count = len(space)
        self.initial_temperature = initial_temperature
        self.max_iter = max_iter
        self.plato_criterion = plato_criterion
        self.log_directory = log_directory
        self.objective = objective
        self.task_name = task_name
        
        if log_directory != '':
            self.search_history = os.path.join(self.log_directory, 'search_history.csv')
        else:
            self.search_history = ''
    
    def save_log_to_file(self, iteration = -1):
        if self.log_directory == '':
            return
        
        if not os.path.exists(self.log_directory):
            # create dir
            os.mkdir(self.log_directory) 
        if not os.path.exists(self.search_history):
            # create metrics_file
            with open(self.search_history, 'w') as f:
                f.write( 'task_name;iteration;' +';'.join(self.space.keys())+';objective\n')
                
        with open(self.search_history, 'a') as f:
            f.write( f'{self.task_name};{iteration};' + ';'.join(np.array(self.best_state, dtype=str))+f';{self.best_objective}\n')            
        
        
    def random_move(self):
        rand_dim = random.randint(0, self.dim_count-1)
        rand_direction = random.randint(0, 1) * 2 - 1
        new_state = self.best_state.copy()
        new_state[rand_dim] = (new_state[rand_dim] + rand_direction)%self.ranges[rand_dim]
        return new_state
    
    def calc_current_temperature(self, iteration_number):
        return self.initial_temperature / float(iteration_number/5 + 1)
    
    def criterion(self, new):
        return math.exp( - (new - self.best_objective) / self.current_temperature)
    
    def run(self):
        self.best_state = self.initial_state
        self.best_objective = self.calc_objective(self.initial_state)
        self.save_log_to_file(0)
        self.state_log = [self.best_state]
        self.objective_log = [self.best_objective]
        self.last_improvement_iteration = 0
        for iteration in range(self.max_iter):
            self.current_temperature = self.calc_current_temperature(iteration)
            new_state = self.random_move()
            new_objective = self.calc_objective(new_state)
            if new_objective < self.best_objective:
                self.best_state, self.best_objective = new_state[:], new_objective
                self.last_improvement_iteration = iteration                
                print(f"Step: {iteration}, Improvement, Objective: {self.best_objective}, State: {self.best_state}")
            else:
                if random.random() < self.criterion(new_objective):
                    self.best_state, self.best_objective = new_state[:], new_objective
                    print(f"Step: {iteration}, Objective: {self.best_objective}, State: {self.best_state}")
            self.state_log.append(self.best_state[:])
            self.objective_log.append(self.best_objective)
            self.save_log_to_file(iteration+1)
            if iteration >= self.last_improvement_iteration + self.plato_criterion:
                break           
    
    def calc_objective(self, state):
        print(state)
        return self.objective(self.space, state)