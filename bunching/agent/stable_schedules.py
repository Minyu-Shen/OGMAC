import wandb

api = wandb.Api()
# put your wandb login name here
entity = 'samuel'


def find_stable_H_and_schedules(analy_name, alpha):
    hyper_paras = {'alpha': alpha, 'behav_polic': analy_name+'_UPDATE'}
    filt = {'config.' + k: v for k, v in hyper_paras.items()}
    runs = api.runs(entity + '/update', filters=filt)
    assert len(runs) <= 1, 'no more than one run found'

    histr = runs[0].history()
    last_Hs = histr['H'].tolist()[-10:]
    H = sum(last_Hs) / len(last_Hs)
    if analy_name == 'XUAN':
        bus_stop_trip_depar_dict = histr['bus_stop_trip_depar_dict'].tolist(
        )[-1]
        bus_stop_trip_depar_dict = eval(bus_stop_trip_depar_dict)
        return H, bus_stop_trip_depar_dict
    elif analy_name == 'NONLINEAR':
        return H, None


if __name__ == '__main__':
    H, sched = find_stable_H_and_schedules('XUAN', 0.5)
    print(H, sched)
