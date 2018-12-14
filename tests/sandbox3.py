import pandas

ds = pandas.read_csv("data/synth-easy2.csv")

ds0 = pandas.get_dummies(ds, columns=["s","y"])

ds0[["y_0","y_1"]].sum(axis=0)

ds0.to_csv("experiments/synth_easy2_3/representations/original_non_scaled.csv", index=False)