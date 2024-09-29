from collections import OrderedDict

def dynamic_EMA(args, aggregated_weights, global_model, beta=0.5):
    ema_weights = OrderedDict()
    
    for key in aggregated_weights.keys():
        ema_weights[key] = beta * aggregated_weights[key] + (1.0 - beta) * global_model[key]
        # print("key : ", key, ema_weights[key].device)
        
    # for ema_param, model_param in zip(aggregated_weights, global_model):
    #     ema_param.data.mul_(beta).add_(model_param.data, alpha=1.0-beta)

    return ema_weights