import os

import numpy as np
import pandas as pd

from .. import constants as con
from ..sequences import generate_dataframes
from ..utils import isoelastic_utility, wealth_change


def sigmoid(deu, beta):
    sdeu = -1 * beta * deu
    return 1 / (1 + np.exp(sdeu))

def simulate_agent(agent:str='0.0x0.0',
                   lambd:float=0.0,
                  eta:float=0.0,
                  mode:int=1,
                  c_dict=con.c_dict,
                  assymetry_dict=con.assymetry_dict,
                  LIMITS:dict=con.LIMITS,
                  filtering=False,
                  n_trials:int=1000,
                  log_beta:dict={0.0: -2, 1.0: 4}):

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
                                  'agent','trial']}

    wealth = 1000
    for trial in range(n_trials):
        logDict = {}
        if mode == 3: #train_tracks
            if wealth  > LIMITS[lambd][1]:
                this_trial = active['bad'].iloc[trial].to_dict()
                track = 'bad'
            elif wealth < LIMITS[lambd][0]:
                this_trial = active['good'].iloc[trial].to_dict()
                track = 'good'
            else:
                this_trial= active['neutral'].iloc[trial].to_dict()
                track = 'neutral'
        else:
            this_trial= active['neutral'].iloc[trial].to_dict()
            track = 'neutral'

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
            u_i = [isoelastic_utility(d_wealth,eta).item() for d_wealth in delta_wealths]
            u = isoelastic_utility(wealth, eta)
            du = [i - u for i in u_i]
            deu = ((du[0] + du[1]) / 2) - ((du[2] + du[3]) / 2)
            choice_probability = sigmoid(deu, np.exp(log_beta[eta]))
            choice = 'left' if choice_probability > np.random.uniform(0,1) else 'right'

        if choice == 'left':
            ch_gamma = current_gammas[:2][np.abs(coin_toss -1)]
            logDict.update({'realized_gamma': ch_gamma})
        elif choice == 'right':
            ch_gamma = current_gammas[2:][np.abs(coin_toss -1)]
            logDict.update({'realized_gamma': ch_gamma})

        wealth = wealth_change(wealth, ch_gamma, lambd).item()

        logDict.update({'track': track, 'gamma_left_up': gamma1, 'gamma_left_down': gamma2,
                        'gamma_right_up': gamma3, 'gamma_right_down': gamma4,
                        'selected_side': choice, 'wealth': wealth, 'event_type': 'WealthUpdate',
                        'eta': lambd, 'agent': agent, 'trial':trial})

        for ld in logDict.keys():
            df[ld].append(logDict[ld])

    df = pd.DataFrame(df)

    return df

if __name__ == '__main__':

    log_beta = {0.0: -1, 0.5: 1, 1.0: 4}
    mode = 3
    lambds = [0,1]
    etas = [[0.0,0.0],[0.0,0.5],[0.0,1.0],
            [0.5,0.0],[0.5,0.5],[0.5,1.0],
            [1.0,0.0],[1.0,0.5],[1.0,1.0]]

    save_path = os.path.join(os.path.join(os.path.dirname(__file__),),'..','..', 'data','outputs','simulations',f'version_{str(mode)}')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for c, lambd in enumerate(lambds):
        for i, agent in enumerate(etas):
            eta = etas[i][c]
            agent_type = f'{agent[0]}x{agent[1]}'

            df = simulate_agent(agent = agent_type, lambd = lambd, eta = eta, mode = mode, log_beta = log_beta)
            df.to_csv(os.path.join(save_path,f'sim_agent_{agent_type}_lambd_{c}.csv'),sep='\t')
