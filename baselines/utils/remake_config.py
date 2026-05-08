def clean_model_config(config, keys_to_remove):
    cfg = config.copy()

    for k in keys_to_remove:
        cfg.pop(k, None)

    return cfg