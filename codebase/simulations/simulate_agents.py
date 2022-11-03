import os

import numpy as np
import pandas as pd

from .. import constants as con
from ..sequences import generate_dataframes
from ..utils import isoelastic_utility, wealth_change


def simulate_agent(lambd:float=0.0,
                  eta:float=0.0,
                  mode:int=1,
                  c_dict=con.c_dict,
                  assymetry_dict=con.assymetry_dict,
                  LIMITS:dict = con.LIMITS,
                  filtering=False,
                  n_trials:int=160):

    seqs = generate_dataframes(lambd=lambd,
                               mode=mode,
                               n_trials_active=n_trials,
                               n_trials_passive_before_reset=45,
                               n_resets_passive=4,
                               c_dict=c_dict,
                               assymetry_dict=assymetry_dict,
                               gamble_filter=filtering)[1]

    if mode == 3:
        active = {jj: kk for jj, kk in  zip(['bad','neutral','good'], seqs)}
    else:
        active = {'neutral': seqs}

    df = {kk : [] for kk in ['event_type', 'selected_side',
                                  'gamma_left_up','gamma_left_down',
                                  'gamma_right_up', 'gamma_right_down',
                                  'track', 'wealth', 'realized_gamma', 'eta',
                                  'simulation_eta','trial']}

    wealth = 1000
    for trial in range(n_trials):
        logDict = {}
        if mode == 3: #train_tracks
            if wealth  > LIMITS[lambd][1]:
                this_trial = active['bad'].iloc[trial].to_dict()
                logDict.update({'track': 'bad'})
            elif wealth < LIMITS[lambd][0]:
                this_trial = active['good'].iloc[trial].to_dict()
                logDict.update({'track': 'good'})
            else:
                this_trial= active['neutral'].iloc[trial].to_dict()
                logDict.update({'track': 'neutral'})
        else:
            this_trial= active['neutral'].iloc[trial].to_dict()
            logDict.update({'track': 'neutral'})


        if this_trial != None:
            gamma1, gamma2 = this_trial['gamma_left_up'], this_trial['gamma_left_down']
            gamma3, gamma4 = this_trial['gamma_right_up'], this_trial['gamma_right_down']
            lambd = this_trial['lambda']
            coin_toss = np.int32(this_trial['gamble_up'])

        current_gammas = [gamma1, gamma2, gamma3, gamma4]

        delta_wealths = [wealth_change(wealth,gamma,lambd).item() for gamma in current_gammas]

        if any(delta_wealths) < 0:
            choice = 0 #Dummy value that is deleted before inference is done
        else:
            u = [isoelastic_utility(d_wealth,eta).item() for d_wealth in delta_wealths]
            choice = np.mean([u[0],u[1]]) > np.mean([u[2],u[3]])

        #introduce some randomness
        choice = not choice if np.random.rand() > 0.8 else choice

        if choice:
            response = 'left'
        else:
            response = 'right'

        if response == 'left':
            ch_gamma = current_gammas[:2][np.abs(coin_toss -1)]
            logDict.update({'realized_gamma': ch_gamma})
        elif response == 'right':
            ch_gamma = current_gammas[2:][np.abs(coin_toss -1)]
            logDict.update({'realized_gamma': ch_gamma})

        wealth = wealth_change(wealth, ch_gamma, lambd).item()

        logDict.update({'gamma_left_up': gamma1, 'gamma_left_down': gamma2,
                        'gamma_right_up': gamma3, 'gamma_right_down': gamma4,
                        'selected_side': response, 'wealth': wealth, 'event_type': 'WealthUpdate',
                        'eta': lambd, 'simulation_eta': eta, 'trial':trial})

        for ld in logDict.keys():
            df[ld].append(logDict[ld])

    df = pd.DataFrame(df)

    return df

if __name__ == '__main__':

    mode = 3
    n_agents = 15
    etas = np.array([np.full(shape=5,fill_value=0.0,dtype=float),
                     np.full(shape=5,fill_value=0.5,dtype=float),
                     np.full(shape=5,fill_value=1.0,dtype=float),]).flatten()

    save_path = os.path.join(os.path.join(os.path.dirname(__file__),),'..','..', 'data','outputs','simulations',f'version_{str(mode)}')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for lambd in [0.0,1.0]:
        for agent in range(n_agents):
            df = simulate_agent(lambd = lambd, eta = etas[agent], mode = mode)
            df.to_csv(os.path.join(save_path,f'sim_agent_{agent}_lambd_{lambd}.csv'),sep='\t')
