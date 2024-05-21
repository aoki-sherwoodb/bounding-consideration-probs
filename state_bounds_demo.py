from calculate_bounds import get_state_consideration_prob_bounds

'''
A script demonstrating the derivation and propagation of bounds on item consideration probabilities 
in a Plackett-Luce plus consideration (PL+C) model framework. 

The universe of items consists of the 50 U.S. states.
'''

alpha = 5
k = 3
df = get_state_consideration_prob_bounds(alpha, k)

print(df)
