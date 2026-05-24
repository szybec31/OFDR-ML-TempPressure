def build_groups(df, leave_one_condition_out = False):

    if not leave_one_condition_out:
        return df["dT"]

    else:
        return (
            "DT" + df["dT"].astype(str) +
            "_P" + df["pressure"].astype(str)
        )