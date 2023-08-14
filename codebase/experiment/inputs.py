import os
from ..file_handler import make_filename
from ..sequences import generate_dataframes


def run(lambd:float, n_resets_passive:int, n_trials_passive_before_reset:int,
        n_trials_active:int, save_path:str, mode:int = 1,
        gamble_filter:bool = False):

    if mode == 1:
        p_df, a_df, meta = generate_dataframes(lambd=lambd,
                                            n_trials_active=n_trials_active,
                                            n_resets_passive=n_resets_passive,
                                            n_trials_passive_before_reset=n_trials_passive_before_reset,
                                            mode=mode,
                                            gamble_filter=gamble_filter
                                            )

        p_df.to_csv(save_path.replace('meta', 'passive').replace('txt', 'tsv').replace('_neutral', ''), index=False, sep='\t')
        a_df.to_csv(save_path.replace('meta', 'active').replace('txt', 'tsv').replace('_neutral', ''), index=False, sep='\t')
        with open(save_path,"w+") as f:
            f.writelines(meta)

    elif mode == 3:
        p_df, a_df = generate_dataframes(lambd=lambd,
                                            n_trials_active=n_trials_active,
                                            n_resets_passive=n_resets_passive,
                                            n_trials_passive_before_reset=n_trials_passive_before_reset,
                                            mode=mode,
                                            gamble_filter=gamble_filter
                                            )

        p_df.to_csv(save_path.replace('meta', 'passive').replace('txt', 'tsv').replace('_neutral', ''), index=False, sep='\t')
        for i, name in enumerate(['bad','neutral','good']):
            a_df[i].to_csv(save_path.replace('meta', 'active').replace('txt', 'tsv').replace('neutral', name), index=False, sep='\t')

    else:
        raise ValueError("Mode has to be 1 or 3")


def run_with_dict(expInfo):

    save_path = make_filename('data/inputs/', expInfo['participant'], expInfo['session'],
                              expInfo['eta'], 'meta', None, 'input_neutral.txt')

    reply = True

    if os.path.isfile(save_path):
        from psychopy import gui

        dlg = gui.Dlg(title="File exists!")
        dlg.addText(f'{save_path} already exitsts!')
        dlg.addField('Overwrite', choices=[True, False])
        reply = dlg.show()

        reply = (reply[0] and dlg.OK)

    if reply:
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)

        run(lambd=expInfo['eta'],
            n_resets_passive=expInfo['n_resets_passive'],
            n_trials_passive_before_reset=expInfo['n_trials_passive_before_reset'],
            n_trials_active=expInfo['n_trials_active'],
            save_path=save_path,
            mode=expInfo['mode'],
            gamble_filter=expInfo['gambleFilter'])

    else:
        print(f"Not creating new inputs for participant {expInfo['participant']}")
        pass
