#!/usr/bin/env python
# coding: utf-8



get_ipython().system(' ./setup.sh')




get_ipython().run_line_magic('matplotlib', 'inline')



import numpy as np
import pandas as pd
from functools import partial
from tqdm.notebook import tqdm
from covidsimulation import plot, run_simulations, SimulationConstants
from covidsimulation.age_group import AgeGroup
from covidsimulation.calibrate import get_best_random_states,    score_reported_deaths
from covidsimulation.disease_parameters import OUTCOME_THRESHOLDS
from covidsimulation.intervention import DiagnosisDelayChange,    SocialDistancingChange
from covidsimulation.parameters import Parameters
from covidsimulation.population import Population
from covidsimulation.random import LogUniformParameter, TriangularParameter,    UniformParameter




### Import Data
casesRawData = pd.read_csv('coviddata/cases.csv')
#casesRawData.head()
deathsRawData = pd.read_csv('coviddata/deaths.csv')
recoveredRawData = pd.read_csv('coviddata/recovered.csv')




age_structure = {  #Estimate US Age distribution from 2019
    '0-9': 0.13,
    '10-19': 0.15,
    '20-29': 0.15,
    '30-39': 0.15,
    '40-49': 0.13,
    '50-59': 0.12,
    '60-69': 0.09,
    '70-79': 0.05,
    '80+': 0.03,
}

assert sum(age_structure.values()) == 1.0 #sum of age_structure =1


USAgeGroup = partial(
    AgeGroup,
    diagnosis_delay=10.0,
    chance_of_diagnosis_if_moderate=0.0,
    isolation_effectiveness=0.9,
)

ISOLATION_PROPENSITIES = [ #probabilites from going out(-1) to stay in(2)
    1.0,  # 0–9 Children have little reason to go out
    1.0,  # 10–19
    -0.5,  # 20–29 Young people are less concerned about the risks
    -0.5,  # 30–39
    0.0,  # 40–49
    0.3,  # 50–59
    0.8,  # 60–69
    1.8,  # 70–79 Eldery will isolate well
    1.8,  # 80+
]

age_groups = [
    USAgeGroup(
        i,
        OUTCOME_THRESHOLDS[i],  # disease severity
        ISOLATION_PROPENSITIES[i],
    ) for i, nome_faixa in enumerate(age_structure.keys())
]

total_inhabitants = 330000000  # 330 million

# 25% of people live alone, 28% of houses have 2 people, 
home_size_probabilities = np.array([0.25, 0.28, 0.27, 0.15, 0.05])
assert home_size_probabilities.sum() == 1.0

all_population = Population(
    name='all',
    age_probabilities=np.array(list(age_structure.values())),
    age_groups=age_groups,
    home_size_probabilities=home_size_probabilities,
    inhabitants=total_inhabitants,
    isolation_propensity_increase=TriangularParameter(
        'isolation_uncertainty',
        -0.5,
        0.0,
        0.5,
    ),  # don't know real isolation 
    seed_infections=10,
    geosocial_displacement=0.0,  
)




interventions = [ #adjustable 
 SocialDistancingChange('2020-03-12', 0.3),  
 SocialDistancingChange('2020-03-22', 0.9),  # stay-at-home order
 DiagnosisDelayChange('2020-04-13', 7.0),  # More testing introduced
 DiagnosisDelayChange('2020-04-15', 5.0),
 SocialDistancingChange('2020-03-16', 0.4),
 SocialDistancingChange('2020-03-22', 0.68),
 SocialDistancingChange('2020-03-29', 0.66),
 SocialDistancingChange('2020-04-05', 0.62),
 SocialDistancingChange('2020-04-12', 0.60),
 SocialDistancingChange('2020-04-24', 0.55),
 SocialDistancingChange('2020-05-02', 0.59),
 SocialDistancingChange('2020-05-16', 0.53),
 SocialDistancingChange('2020-05-30', 0.62),
 SocialDistancingChange('2020-06-06', 0.57),
 SocialDistancingChange('2020-06-13', 0.54),
 SocialDistancingChange('2020-06-27', 0.48),
 SocialDistancingChange('2020-07-04', 0.43),
 SocialDistancingChange('2020-07-18', 0.41),
 SocialDistancingChange('2020-07-25', 0.53),
 SocialDistancingChange('2020-08-08', 0.55),
 SocialDistancingChange('2020-08-22', 0.57),
 DiagnosisDelayChange('2020-04-06', 10.0),  
 DiagnosisDelayChange('2020-04-15', 8.0),
 DiagnosisDelayChange('2020-04-22', 6.5),
 HygieneAdoption('2020-03-13', TriangularParameter('hygiene_adoption', 0.5, 0.7, 0.9)),#more caution with washing hands
 MaskUsage('2020-03-29', TriangularParameter('mask_adoption', 0.5, 0.7, 0.9) * 0.5),
 MaskUsage('2020-04-24', TriangularParameter('mask_adoption', 0.5, 0.7, 0.9)),
]


params = Parameters(
    [all_population],
    SimulationConstants(),
    interventions=interventions,
    d0_infections=LogUniformParameter('sp_d0_infections', 12000, 50000), #boundarys can be adjusted
    start_date='2020-03-10',
    min_age_group_initially_infected=4,
)


US_official_deaths = [ #used to score the model for accuracy
    ('2020-03-10', 26.0), 
    ('2020-03-20', 150.0), 
    ('2020-03-30', 2509.0), 
    ('2020-04-09', 14817.0), 
    ('2020-04-19', 38910.0),  
    ('2020-04-23', 46784.0),
    ('2020-04-28', 56245.0),
    ('2020-05-05', 68934.0),
    ('2020-05-18', 89562.0),
    ('2020-05-30', 102836.0),
    ('2020-06-07', 109802.0),
    ('2020-06-25', 121979.0),
    ('2020-07-04', 129434.0),
    ('2020-07-18', 139266.0),
    ('2020-08-01', 153314.0),
    ('2020-08-15', 168446.0),
    ('2020-08-29', 181773.0),
]
US_score_function = partial(
    score_reported_deaths,
    expected_deaths=US_official_deaths
)



#MODEL Training:: step 1
states = get_best_random_states(
    score_function=US_score_function,
    sim_params=params,
    random_states=[],
    simulate_capacity=False,
    duration=158, #start to end of model, july 31 +aug
    simulation_size=100000, #100k
    n=100, #100 simulations
    p=0.1, #take best 10 model
    use_cache=True,
    tqdm=tqdm,
)




#MODEL Training:: step 2
states = get_best_random_states(
    score_function=US_score_function,
    sim_params=params,
    random_states=states,
    simulate_capacity=False,
    duration=158, 
    simulation_size=1000000, #100 million
    n=100, #match the p*n from step 1
    p=0.16, #top 16 best model
    use_cache=True,
    tqdm=tqdm,
)




#MODEL Training:: step 3
stats = run_simulations(
    sim_params=params,
    simulate_capacity=False,
    duration=158,
    number_of_simulations=30, # 2*(p*n) from step 2
    simulation_size=330000000, #330 million
    random_states=states,
    tqdm=tqdm,
)




import requests
import plotly.graph_objects as go
r = requests.get('https://api.covidtracking.com/v1/us/daily.json')
b = r.json()
Death = []
for i in range(duration):
    f = str(b[i]['date'])
    die = int(str(b[i]['death']))
    f = f[0:4] + '-' + f[4:6] + '-' + f[6:8]
    Death.append(die)
#print(xDate)



#Accuracy Measure
#model data
def accuracyCheck(upperBoundModel, lowerBoundModel, actual):
    correct = 0
    for i in range(len(actual)):
        if actual[i] >= lowerBoundModel[i] and actual[i] <= upperBoundModel[i]:
            correct = correct + 1
    return float(correct/len(actual))

#alter for other tests
UpBound= stats.get_upperbound('confirmed_deaths')
lowBound = stats.get_lowerbound('confirmed_deaths')
accuracyCheck(UpBound, lowBound, Death)





#plot([
#        (saved_stats.get_metric('infected'), 'Nothing done'),
#        (stats_isolation.get_metric('infected'), 'Social distancing'),
#    ], 'Infected', True, stop=90)


#plot([
#        (stats_isolation.get_metric('deaths'), 'Nothing done'),
#        (stats_isolation.get_metric('confirmed_deaths'), 'Social distancing'),
#    ], 'Infected', False, stop=90)




fig = plot([
    (
        stats.get_metric('confirmed_deaths'),
        'Total Projection'
    ),
   (
        stats.get_metric('confirmed_deaths', 'all', '0-25'),
        '0-25 Age Average',    
   ),
   (
        stats.get_metric('confirmed_deaths', 'all', '65+'),
        '65+ Age Average',    
   ) 
],
    'Projected death count in US',
    False,
)
fig.show()  



fig = plot([
    (
        stats.get_metric('confirmed_deaths'),
        'projection',
    ),
],
    'Projected Daily Deaths infected in US',
    False,
)
fig.show()  




fig = plot([
    (
        stats.get_metric('infected'),
        'projection',
    ),
],
    'Projected Daily Cases in US',
    False,
)
fig.show()  



fig = plot([
    (
        stats.get_metric('hospitalized'),
        'projection Apr/19',
    ),
],
    'Projected Cumulative Hosputalized in US',
    False,
)
fig.show()  





fig = plot([
    (
        stats.get_metric('susceptible'),
        'projection',
    ),
],
    'Projected susceptible in US',
    False,
)
fig.show()  



stats.metrics['fatality_rate'] = ('confirmed_deaths', 'infected')
fig = plot([
    (
        stats.get_metric('fatality_rate'),
        'population',
    ),
    (
        stats.get_metric('fatality_rate', 'all', '60-69'),
        '60-69 only',
    ),
    (
        stats.get_metric('fatality_rate', 'all', '20-29'),
        '20-29 only',
    ),
],
    'Projected observed fatality rate',
    False,
)
fig.show()  





