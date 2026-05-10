"""
LightGBM Training + Upgraded Recommendation Demo

Run: python demo_lgbm.py
"""
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.recommender.train_lgbm import (
    build_training_data, train, save_model, load_model, predict_opportunities
)
from src.recommender.scorer import _generate_action

console = Console()
TARGET_BRAND = "Asana"


def print_feature_importance(model, features: list[str]):
    importance = pd.Series(model.feature_importances_, index=features)
    table = Table(title="Feature Importance")
    table.add_column("Feature", style="cyan")
    table.add_column("Importance", justify="right")
    for feat, imp in importance.sort_values(ascending=False).items():
        table.add_row(feat, str(round(float(imp), 1)))
    console.print(table)


def print_recommendations(result_df: pd.DataFrame):
    table = Table(title=f"LightGBM Recommendations for '{TARGET_BRAND}'")
    table.add_column("Domain Type", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Your Citations", justify="right")
    table.add_column("Competitor Avg", justify="right")
    table.add_column("Action")

    priority_style = {True: "red", False: "yellow"}
    for _, row in result_df.head(8).iterrows():
        is_high = row["opportunity_score"] >= 60
        score_style = "red" if is_high else "yellow"
        action = _generate_action(row["domain_type"], 0, 0)
        table.add_row(
            row["domain_type"],
            f"[{score_style}]{row['opportunity_score']}[/]",
            str(int(row["cooccurrence"])),
            str(round(float(row["competitor_avg_cooccurrence"]), 1)),
            action[:55] + "..." if len(action) > 55 else action,
        )
    console.print(table)


def main():
    # 1. Build features
    console.print("\n[bold]Step 1: Building feature matrix from BigQuery...[/]")
    df = build_training_data()
    console.print(f"  Rows: {len(df)}  Brands: {df['brand'].nunique()}  "
                  f"Domain types: {df['domain_type'].nunique()}")

    # 2. Train
    console.print("\n[bold]Step 2: Training LightGBM...[/]")
    model, le, features = train(df)

    # 3. Save
    save_model(model, le, features)

    # 4. Feature importance
    console.print("\n[bold]Step 3: Feature Importance[/]")
    print_feature_importance(model, features)

    # 5. Predict for target brand
    console.print(f"\n[bold]Step 4: LightGBM Recommendations for '{TARGET_BRAND}'[/]")
    result = predict_opportunities(model, features, le, TARGET_BRAND, df)
    if result.empty:
        console.print(f"[yellow]No data for {TARGET_BRAND} — try another brand[/]")
    else:
        print_recommendations(result)

    # 6. Show available brands
    console.print(f"\n[dim]Available brands: {sorted(df['brand'].unique().tolist())}[/]")


if __name__ == "__main__":
    main()
