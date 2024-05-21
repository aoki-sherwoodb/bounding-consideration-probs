import numpy as np
import pandas as pd
from torch import tensor

from data_utils import get_random_10, get_top_3, get_state_dicts, topo_sort, get_topk_probs, get_topk_swaps
from logit import Logit

def get_lb_baselines(alpha: int, k: int, choices_list: list) -> list:
    '''
    Derive baseline lower bounds on consideration probabilities.
    '''
    alpha_k_coef = 1 - (alpha * np.exp(1 - alpha))**k
    lower_bounds = [choices_list.count(i) * alpha_k_coef / (len(choices_list) / 3) for i in range(50)]  # from Thm 4.2
    
    return lower_bounds


def tighten_lb(lb_baseline: list, k: int, topk_swaps: dict, topk_probs: list) -> list:
    '''
    Tighten lower bounds on consideration probabilities.
    '''
    b = lb_baseline.copy()
    topo = topo_sort(topk_swaps)
    for i in topo:
        for j in topk_swaps[i]:
            for l in range(1, k + 1):
                topl_probs = topk_probs[l - 1]
                c = topl_probs[i] / topl_probs[j]
                if c > 0.0:  # avoid division by 0 errors
                    b[j] = max(b[j], b[i] / (c - (c * b[i]) + b[i]))
    return b


def get_ub_baselines(utilities: tensor, alpha: int, k: int, choices_list: list) -> list:
    '''
    Derive baseline upper bounds on consideration probabilities.
    '''
    n = len(utilities) 

    additive_term = (k * (alpha * np.exp(1 - alpha))**k) / (1 - (alpha * np.exp(1 - alpha))**k)
    sum_exp_utils = np.sum(np.exp(utilities.detach().numpy()))
    top1_probs = [0] * 50

    for i in range(0, len(choices_list), 3):
        top1_probs[choices_list[i]] += 1

    top1_probs = np.divide(top1_probs, len(choices_list) / 3) # normalize for number of 1st places chosen
    upper_bounds = [(sum_exp_utils / np.exp(utilities[i].item())) * (top1_probs[i] + additive_term) for i in range(n)]  # plug into Thm 5.3

    return upper_bounds


def tighten_ub(ub_baseline, k, topk_swaps, topk_probs):
    '''
    Tighten upper bounds on consideration probabilities.
    '''

    b = np.array(ub_baseline)
    np.clip(b, 0, 1, b)

    # reverse swap dict to create adjacency list this time
    adjacency_list = {i: [] for i in range(50)}
    for i in topk_swaps:
        for j in topk_swaps[i]:
            adjacency_list[j].append(i)

    topo = topo_sort(adjacency_list)

    for j in topo:
        for i in adjacency_list[j]:
            for l in range(1, k + 1):

                topl_probs = topk_probs[l - 1]
                c = topl_probs[i] / topl_probs[j]

                if c > 0.0 and c <= 1.0: # protect against divide by 0 errors
                    b[i] = min(b[i], c * b[j] / (1 - b[j] + (c * b[j])))
    return b


def get_state_consideration_prob_bounds(alpha: int = 5, k: int = 3, sort_by: str = 'lower', sort_descending: bool = True) -> pd.DataFrame:
    '''
    Constructs a sorted dataframe of states with baseline and propagated lower and upper bounds on their consideration probabilities.

    @param alpha: expected consideration set size. Defaults to 5.
    @param k: length of rankings from which consider-then-choose data is derived. Defaults to 3.
    @param sort_by: string specifying which (propagated) bounds to sort by. 'lower', 'upper', sort by lower and 
                    upper bounds respectively, and None will sort by state name alphabetically.
    @param sort_descending: whether to sort in descending order of consideration probability bound.

    @return: a sorted dataframe of states with bounds.
    '''
    choice_sets_r10, _, choices_r10 = get_random_10()
    _, _, choices_t3 = get_top_3()
    choices_list = choices_t3.tolist()
    state_id, _, _ = get_state_dicts()
    state_names = state_id.keys()

    model_r10 = Logit(50)
    model_r10.fit(choice_sets_r10, choices_r10)
    r10_utils = model_r10.utilities

    swaps = get_topk_swaps(choices_list, r10_utils, k)
    empirical_ranking_probs = get_topk_probs(choices_list, k)

    # derive baseline bounds on consideration probabilities (see Thms 4.2 & 5.3)
    ub_baseline = get_ub_baselines(r10_utils, alpha, k, choices_list)
    lb_baseline = get_lb_baselines(alpha, k, choices_list)

    # propagate bounds along swaps in PL utilities and empirical top-k ranking probabilities (see Algorithms 1 & 2)
    ub_tight = tighten_ub(ub_baseline, k, swaps, empirical_ranking_probs)
    lb_tight = tighten_lb(lb_baseline, k, swaps, empirical_ranking_probs)

    #states_sorted = sorted(state_names, key=lambda state:sort_bound[state_id[state]], reverse=not sort_descending)

    bounds_sorted = {state: (lb_tight[state_id[state]], ub_tight[state_id[state]], lb_baseline[state_id[state]], ub_baseline[state_id[state]]) for state in state_names}

    df = pd.DataFrame.from_records(bounds_sorted).transpose()
    df = df.rename(columns={0: "lower bound final", 1: "upper bound final", 2: "lower bound baseline", 3: "upper bound baseline"})
    df = df.reset_index(names="State")
    if sort_by == 'lower':
        df = df.sort_values(by='lower bound final', ascending=not sort_descending)
    elif sort_by == 'upper':
        df = df.sort_values(by='upper bound final', ascending=not sort_descending)
    
    return df

