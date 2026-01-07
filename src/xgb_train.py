import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss


def start_training(data_path):
    df = pd.read_csv(data_path)

    # identify surface columns
    surface_cols = [col for col in df.columns if col.startswith('surf_')]

    # only use numeric features for baseline
    features = ['p1_rank', 'p2_rank', 'p1_gen_elo', 'p2_gen_elo', 'p1_surf_elo', 'p2_surf_elo'] + surface_cols
    X = df[features]
    y = df['target']

    acc_scores = []
    ll_scores = []

    # for each time series fold
    tss = TimeSeriesSplit(n_splits=5)
    for i, (train_index, test_index) in enumerate(tss.split(X)):
        # split
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # train
        model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
        model.fit(X_train, y_train)

        # evaluate
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)

        # log loss
        test_probs = model.predict_proba(X_test)
        ll = log_loss(y_test, test_probs)

        acc_scores.append(test_acc)
        ll_scores.append(ll)
        print(f"Fold {i+1} - Train Acc: {train_acc:.2%}, Test Acc: {test_acc:.2%}")
    
    print(f"\nAverage Cross-Validation Accuracy: {sum(acc_scores)/len(acc_scores):.2%}")
    print(f"Avg Log Loss: {sum(ll_scores)/len(ll_scores):.4f}")
    xgb.plot_importance(model)
    plt.show()

    # save baseline model
    # model.save_model("data/processed/baseline_model.json")
    # to load it back later:
    # loaded_model = xgb.XGBClassifier()
    # loaded_model.load_model("data/processed/baseline_model.json")

    return model, test_acc

if __name__ == "__main__":
    from pathlib import Path
    test_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "atp_with_elo.csv"
    if test_path.exists():
        start_training(test_path)
    else:
        print("Processed file not found. Run process.py first.")