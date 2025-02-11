from collections import OrderedDict

def MADA(args, aggregated_weights, global_model, beta=0.5):
    ema_weights = OrderedDict()
    
    for key in aggregated_weights.keys():
        ema_weights[key] = beta * aggregated_weights[key] + (1.0 - beta) * global_model[key]

    return ema_weights