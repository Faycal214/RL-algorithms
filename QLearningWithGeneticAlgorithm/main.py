import numpy as np
import random
from tqdm import tqdm

# env = gym.make('Taxi-v3', render_mode = 'rgb_array')
# print(env.observation_space.n)
# print(env.action_space.n)

class Genetic_algo :
    def __init__(self, nb_individus, nb_generation, env) :
        self.nb_individus = nb_individus
        self.nb_generation = nb_generation
        self.env = env
        self.stateDimension = self.env.observation_space.n
        self.actionDimension = self.env.action_space.n
        self.initiale_population = self.create_population()
        self.mutation_rate = 0.9
        self.mutation_factor = 0.4
    
    def create_population(self) :
        np.random.seed(42)
        population = []
        for _ in range(self.nb_individus) :
            q_table = np.random.uniform(1, 25, (self.stateDimension, self.actionDimension))
            population.append(q_table)
        return population
    
    def evaluate(self, q_table) :
        rewards = 0
        (current_state, _) = self.env.reset()
        done = False
        while not done :
            action = np.argmax(q_table[current_state])
            (next_state, reward, done, _, _) = self.env.step(action)
            rewards += reward
            current_state = next_state
        return rewards
    
    def cross_over(self, parent1, parent2) :
        cross_over_point = int(self.stateDimension / 2)
        enfant1 = np.zeros((self.stateDimension, self.actionDimension))
        enfant2 = np.zeros((self.stateDimension, self.actionDimension))

        enfant1[:cross_over_point, :] = parent1[:cross_over_point, :]
        enfant2[:cross_over_point, :] = parent2[:cross_over_point, :]

        enfant1[cross_over_point :, :] = parent2[cross_over_point :, :]
        enfant2[cross_over_point :, :] = parent1[cross_over_point :, :]

        return enfant1, enfant2
    
    def generation_creation(self, population) :
        new_generation = []
        for _ in range(self.nb_individus) :
            parent1, parent2 = random.sample(population, k= 2)
            enfant1, enfant2 = self.cross_over(parent1, parent2)
            new_generation.extend([parent1, parent2, enfant1, enfant2])
        
        total_rewards = []
        for q_table in new_generation :
            rewards = self.evaluate(q_table)
            total_rewards.append(rewards)
        
        solutions_scores = []
        for q_table, reward in zip(new_generation, total_rewards) :
            x = {}
            x['solution'] = q_table
            x['rewards'] = reward
            solutions_scores.append(x)
        
        return solutions_scores

    def selection(self, generation) :
        best_solutions = sorted(generation, key= lambda x: x['rewards'], reverse= True)
        best_solutions = best_solutions[:self.nb_individus]
        q_tables = [solution["solution"] for solution in best_solutions]
        return q_tables
    
    def mutation(self, generation) :
        nb_mutated_solutions = int(self.mutation_rate * self.nb_individus)
        mutated_solutions = random.sample(generation, k= nb_mutated_solutions)

        if self.nb_generation > 2000 :
            for q_table in mutated_solutions :
                for state in range(self.stateDimension) :
                    for action in range(self.actionDimension) :
                        if np.random.rand() > self.mutation_rate :
                            mutation = self.mutation_factor * q_table[state][action]
                            q_table[state][action] += mutation
        
        best_q_table = generation[0]
        reward_best_q_table = self.evaluate(best_q_table)
        for q_table in generation :
            if self.evaluate(q_table) > reward_best_q_table :
                best_q_table = q_table
                reward_best_q_table = self.evaluate(q_table)
        return best_q_table, reward_best_q_table
    
    
    def training_process(self) :
        population = self.create_population()

        for _ in tqdm(range(self.nb_generation), desc="Loading %", miniters= self.nb_generation)  :
            if _ == 0 :
                generation = self.generation_creation(population)
            else :
                generation = self.generation_creation(generation)
            
            generation = self.selection(generation)
            best_q_table, reward = self.mutation(generation)
            if reward != 0 :
                break
        return best_q_table, reward
        

