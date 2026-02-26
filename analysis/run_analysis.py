"""
SKIDGE Full Analysis Script
==========================
Follows the corrected 12-step regression methodology from SKIDGE(1).ipynb
Performs all statistical tests, ML models, and generates visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# =============================================
# CONFIG
# =============================================
DATA_DIR = "data"
IMG_DIR = "analysis-images"
CSV_FILE = os.path.join(DATA_DIR, "Engineer Grad Prof. Outcomes Analysis.csv")
LDJSON_DIR = os.path.join(DATA_DIR, "marketing_sample_for_indeed_co_in-indeed_co_in_job_data__20211001_20211231__30k_data.ldjson")
LDJSON_FILE = os.path.join(LDJSON_DIR, "marketing_sample_for_indeed_co_in-indeed_co_in_job_data__20211001_20211231__30k_data.ldjson")

os.makedirs(IMG_DIR, exist_ok=True)

results = {}  # Store all numeric results for the HTML

# =============================================
# PART A: GRADUATE OUTCOMES ANALYSIS
# =============================================
print("=" * 60)
print("PART A: Engineering Graduate Outcomes Analysis")
print("=" * 60)

# STEP 1-2: Load Data
df = pd.read_csv(CSV_FILE)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
results['n_graduates'] = len(df)
results['n_features'] = df.shape[1]

# STEP 3: Remove Data Leakage Columns
leakage_cols = ["Student_ID", "CollegeID", "CollegeCityID"]
df = df.drop(columns=[c for c in leakage_cols if c in df.columns], errors='ignore')
print(f"After removing leakage columns: {df.shape}")

# STEP 4: Convert DateOfBirth to Age
if 'DateOfBirth' in df.columns:
    df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], errors='coerce')
    df['Age'] = 2024 - df['DateOfBirth'].dt.year
    df = df.drop(columns=['DateOfBirth'])
    print(f"Age created. Range: {df['Age'].min()} - {df['Age'].max()}")

# =============================================
# EXPLORATORY DATA ANALYSIS
# =============================================
print("\n--- Exploratory Statistics ---")

# Key numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")

# Salary statistics
salary_stats = df['Starting_Salary'].describe()
print(f"\nSalary Statistics:")
print(salary_stats)
results['salary_mean'] = round(df['Starting_Salary'].mean(), 2)
results['salary_median'] = round(df['Starting_Salary'].median(), 2)
results['salary_std'] = round(df['Starting_Salary'].std(), 2)
results['salary_min'] = round(df['Starting_Salary'].min(), 2)
results['salary_max'] = round(df['Starting_Salary'].max(), 2)
results['salary_skew'] = round(df['Starting_Salary'].skew(), 3)

# GPA statistics
results['gpa_mean'] = round(df['collegeGPA'].mean(), 2)
results['gpa_std'] = round(df['collegeGPA'].std(), 2)

# =============================================
# VISUALIZATION 1: Salary Distribution (raw vs log)
# =============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['Starting_Salary'], bins=50, color='#2196F3', alpha=0.8, edgecolor='white')
axes[0].set_title('Starting Salary Distribution (Raw)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Starting Salary (₹)')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['Starting_Salary'].mean(), color='red', linestyle='--', label=f'Mean: ₹{df["Starting_Salary"].mean():,.0f}')
axes[0].axvline(df['Starting_Salary'].median(), color='green', linestyle='--', label=f'Median: ₹{df["Starting_Salary"].median():,.0f}')
axes[0].legend()

df_pos = df[df['Starting_Salary'] > 0]
log_salary = np.log(df_pos['Starting_Salary'])
axes[1].hist(log_salary, bins=50, color='#4CAF50', alpha=0.8, edgecolor='white')
axes[1].set_title('Log(Starting Salary) Distribution', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Log(Starting Salary)')
axes[1].set_ylabel('Frequency')
axes[1].axvline(log_salary.mean(), color='red', linestyle='--', label=f'Mean: {log_salary.mean():.2f}')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '01_salary_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# Test normality of log-transformed salary
stat_shapiro, p_shapiro = stats.shapiro(log_salary.sample(min(5000, len(log_salary)), random_state=42))
stat_dagostino, p_dagostino = stats.normaltest(log_salary)
results['log_salary_shapiro_stat'] = round(stat_shapiro, 4)
results['log_salary_shapiro_p'] = f"{p_shapiro:.4e}"
results['log_salary_dagostino_stat'] = round(stat_dagostino, 4)
results['log_salary_dagostino_p'] = f"{p_dagostino:.4e}"
results['log_salary_skew'] = round(log_salary.skew(), 4)

print(f"\nLog(Salary) Shapiro-Wilk: stat={stat_shapiro:.4f}, p={p_shapiro:.4e}")
print(f"Log(Salary) D'Agostino: stat={stat_dagostino:.4f}, p={p_dagostino:.4e}")
print(f"Log(Salary) Skewness: {log_salary.skew():.4f}")

# =============================================
# VISUALIZATION 2: Feature Distributions
# =============================================
key_features = ['collegeGPA', 'HighSchool10thPercentage', 'HighSchool12thPercentage',
                'QuantitativeAptitudeScore', 'LogicalReasoningScore', 'EnglishScore']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for idx, feat in enumerate(key_features):
    if feat in df.columns:
        ax = axes[idx // 3][idx % 3]
        ax.hist(df[feat].dropna(), bins=40, color=sns.color_palette("husl", 6)[idx], alpha=0.8, edgecolor='white')
        ax.set_title(feat, fontsize=11, fontweight='bold')
        ax.axvline(df[feat].mean(), color='red', linestyle='--', alpha=0.7, label=f'μ={df[feat].mean():.1f}')
        ax.legend(fontsize=9)
plt.suptitle('Key Feature Distributions', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '02_feature_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()

# =============================================
# VISUALIZATION 3: Full Correlation Heatmap
# =============================================
numeric_df = df.select_dtypes(include=[np.number])
# Select meaningful columns (exclude any leftover IDs)
corr_cols = [c for c in numeric_df.columns if c not in ['Age']]
corr_matrix = numeric_df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(16, 14))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, square=True, linewidths=0.5, annot_kws={'size': 7},
            vmin=-1, vmax=1)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '03_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()

# Key correlations with salary
salary_corrs = corr_matrix['Starting_Salary'].drop('Starting_Salary').sort_values(ascending=False)
print("\nCorrelations with Starting_Salary:")
print(salary_corrs.head(10))
results['top_salary_correlations'] = {k: round(v, 4) for k, v in salary_corrs.head(10).items()}

# GPA vs Salary correlation
results['gpa_salary_corr'] = round(corr_matrix.loc['collegeGPA', 'Starting_Salary'], 4)
results['gpa_salary_r_squared'] = round(corr_matrix.loc['collegeGPA', 'Starting_Salary'] ** 2, 4)

# =============================================
# VISUALIZATION 4: GPA vs Salary Scatter
# =============================================
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(df['collegeGPA'], df['Starting_Salary'], 
                     alpha=0.3, s=20, c=np.log(df['Starting_Salary'].clip(1)), cmap='viridis')
ax.set_xlabel('College GPA', fontsize=12)
ax.set_ylabel('Starting Salary (₹)', fontsize=12)
ax.set_title('College GPA vs Starting Salary', fontsize=14, fontweight='bold')

# Add correlation annotation
r_val = df['collegeGPA'].corr(df['Starting_Salary'])
ax.annotate(f'Pearson r = {r_val:.3f}\nR² = {r_val**2:.3f}', 
            xy=(0.05, 0.95), xycoords='axes fraction', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
            verticalalignment='top')

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '04_gpa_vs_salary.png'), dpi=150, bbox_inches='tight')
plt.close()

# =============================================
# VISUALIZATION 5: Personality Traits vs Skills/GPA
# =============================================
personality_cols = [c for c in df.columns if 'Score' in c and any(p in c.lower() for p in ['conscient', 'agreeable', 'extraver', 'neurotic', 'open'])]
skill_cols = ['collegeGPA', 'QuantitativeAptitudeScore', 'LogicalReasoningScore', 'EnglishScore', 'ProgrammingSkills']
skill_cols = [c for c in skill_cols if c in df.columns]

if personality_cols and skill_cols:
    corr_personality = df[personality_cols + skill_cols].corr().loc[personality_cols, skill_cols]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_personality, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                ax=ax, linewidths=0.5, vmin=-0.3, vmax=0.3)
    ax.set_title('Personality Traits vs Skills & GPA Correlations', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, '05_personality_vs_skills.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    max_personality_corr = corr_personality.abs().max().max()
    results['max_personality_skill_corr'] = round(max_personality_corr, 4)
    print(f"\nMax |personality-skill correlation|: {max_personality_corr:.4f}")

# =============================================
# STATISTICAL TESTS
# =============================================
print("\n" + "=" * 60)
print("STATISTICAL TESTS")
print("=" * 60)

# 1. One-Sample T-Tests vs Industry Reference (μ = 70)
print("\n--- One-Sample T-Tests vs μ=70 ---")
ttest_features = [c for c in numeric_df.columns if 'Skills' in c or 'Score' in c]
ttest_results = []
for feat in ttest_features:
    data = df[feat].dropna()
    t_stat, p_val = stats.ttest_1samp(data, 70)
    direction = "Above" if data.mean() > 70 else "Below"
    ttest_results.append({
        'feature': feat,
        't_stat': round(t_stat, 3),
        'p_value': f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.6f}",
        'mean': round(data.mean(), 2),
        'direction': direction,
        'significant': p_val < 0.05
    })
    print(f"  {feat}: t={t_stat:.3f}, p={p_val:.4e}, mean={data.mean():.2f}, {direction}")

results['ttest_results'] = ttest_results

# 2. GPA vs Salary: Pearson Correlation + significance
r_gpa_sal, p_gpa_sal = stats.pearsonr(df['collegeGPA'], df['Starting_Salary'])
results['pearson_gpa_salary_r'] = round(r_gpa_sal, 4)
results['pearson_gpa_salary_p'] = f"{p_gpa_sal:.4e}"
print(f"\nPearson (GPA vs Salary): r={r_gpa_sal:.4f}, p={p_gpa_sal:.4e}")

# 3. Spearman Correlation (for skewed salary)
rho, p_spearman = stats.spearmanr(df['collegeGPA'], df['Starting_Salary'])
results['spearman_gpa_salary_rho'] = round(rho, 4)
results['spearman_gpa_salary_p'] = f"{p_spearman:.4e}"
print(f"Spearman (GPA vs Salary): rho={rho:.4f}, p={p_spearman:.4e}")

# 4. Chi-Square Goodness-of-Fit for Categorical Features
print("\n--- Chi-Square Goodness-of-Fit ---")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
chi2_results = []
for col in categorical_cols:
    observed = df[col].value_counts()
    expected = np.full(len(observed), len(df) / len(observed))
    chi2_stat, p_val = stats.chisquare(observed, f_exp=expected)
    chi2_results.append({
        'feature': col,
        'chi2': round(chi2_stat, 1),
        'p_value': f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.6f}",
        'n_categories': len(observed),
        'dominant': observed.idxmax(),
        'dominant_pct': round(observed.max() / observed.sum() * 100, 1)
    })
    print(f"  {col}: χ²={chi2_stat:.1f}, p={p_val:.4e}, dominant={observed.idxmax()} ({observed.max()/observed.sum()*100:.1f}%)")

results['chi2_results'] = chi2_results

# 5. ANOVA — Salary by Categorical Groups
print("\n--- One-Way ANOVA (Salary by Group) ---")
anova_results = []
for col in ['Gender', 'DegreeType', 'MajorSpecialization', 'CollegeState', 'IndustryDomain']:
    if col in df.columns:
        groups = [group['Starting_Salary'].dropna().values for _, group in df.groupby(col) if len(group) > 5]
        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            # Effect size (eta-squared)
            ss_between = sum(len(g) * (np.mean(g) - df['Starting_Salary'].mean())**2 for g in groups)
            ss_total = sum((df['Starting_Salary'] - df['Starting_Salary'].mean())**2)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0
            anova_results.append({
                'factor': col,
                'f_stat': round(f_stat, 3),
                'p_value': f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.6f}",
                'eta_squared': round(eta_sq, 4),
                'significant': p_val < 0.05,
                'n_groups': len(groups)
            })
            print(f"  {col}: F={f_stat:.3f}, p={p_val:.4e}, η²={eta_sq:.4f}")

results['anova_results'] = anova_results

# =============================================
# STEP 5-15: REGRESSION ANALYSIS (CORE)
# =============================================
print("\n" + "=" * 60)
print("ML REGRESSION ANALYSIS (Following SKIDGE(1).ipynb)")
print("=" * 60)

# Prepare data
df_ml = df.copy()
df_ml = df_ml[df_ml['Starting_Salary'] > 0]

# Log transform target
df_ml['Salary_log'] = np.log(df_ml['Starting_Salary'])

# Feature Engineering (from SKIDGE(1).ipynb Step 2)
if 'QuantitativeAptitudeScore' in df_ml.columns:
    df_ml['CognitiveScore'] = (df_ml['QuantitativeAptitudeScore'] + df_ml['LogicalReasoningScore'] + df_ml['EnglishScore']) / 3
if 'HighSchool10thPercentage' in df_ml.columns:
    df_ml['AcademicStrength'] = (df_ml['HighSchool10thPercentage'] + df_ml['HighSchool12thPercentage'] + df_ml['collegeGPA'] * 10) / 3
if 'ProgrammingSkills' in df_ml.columns and 'QuantitativeAptitudeScore' in df_ml.columns:
    df_ml['TechInteraction'] = df_ml['ProgrammingSkills'] * df_ml['QuantitativeAptitudeScore']

y = df_ml['Salary_log']
X = df_ml.drop(columns=['Starting_Salary', 'Salary_log'])

# Identify column types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
print(f"Target: log(Salary), n={len(y)}")

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# =============================================
# MODEL 1: Random Forest Regressor
# =============================================
print("\n--- Random Forest Regressor ---")
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2 = r2_score(y_test, y_pred_rf)

print(f"  MAE (log): {rf_mae:.4f}")
print(f"  RMSE (log): {rf_rmse:.4f}")
print(f"  R² Score: {rf_r2:.4f}")

# Convert back to actual salary
y_pred_actual = np.exp(y_pred_rf)
y_test_actual = np.exp(y_test)
rf_mae_actual = mean_absolute_error(y_test_actual, y_pred_actual)
rf_mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
print(f"  MAE (₹): ₹{rf_mae_actual:,.0f}")
print(f"  MAPE: {rf_mape:.2f}%")

results['rf_mae_log'] = round(rf_mae, 4)
results['rf_rmse_log'] = round(rf_rmse, 4)
results['rf_r2'] = round(rf_r2, 4)
results['rf_mae_actual'] = round(rf_mae_actual, 2)
results['rf_mape'] = round(rf_mape, 2)

# Cross-Validation
cv_scores = cross_val_score(rf_pipeline, X, y, cv=5, scoring='r2', n_jobs=-1)
results['rf_cv_scores'] = [round(s, 4) for s in cv_scores]
results['rf_cv_mean'] = round(cv_scores.mean(), 4)
results['rf_cv_std'] = round(cv_scores.std(), 4)
print(f"  5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Individual folds: {[round(s,4) for s in cv_scores]}")

# Feature Importance
rf_standalone = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
X_processed = preprocessor.fit_transform(X)
rf_standalone.fit(X_processed, y)

# Get feature names
ohe_cats = list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))
feature_names = numeric_features + ohe_cats
importances = rf_standalone.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 15 Feature Importances (RF):")
print(importance_df.head(15).to_string(index=False))
results['rf_top_features'] = importance_df.head(15).to_dict('records')

# VISUALIZATION 6: Feature Importance
fig, ax = plt.subplots(figsize=(12, 8))
top20 = importance_df.head(20)
colors = ['#FF5722' if imp > 0.05 else '#2196F3' for imp in top20['Importance']]
ax.barh(range(len(top20)-1, -1, -1), top20['Importance'].values, color=colors, edgecolor='white')
ax.set_yticks(range(len(top20)-1, -1, -1))
ax.set_yticklabels(top20['Feature'].values, fontsize=9)
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Random Forest Regressor - Top 20 Feature Importances\n(Predicting log(Starting Salary))', fontsize=13, fontweight='bold')
for i, (imp, feat) in enumerate(zip(top20['Importance'].values, top20['Feature'].values)):
    ax.text(imp + 0.002, len(top20)-1-i, f'{imp:.3f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '06_rf_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()

# VISUALIZATION 7: Predicted vs Actual (log scale)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(y_test, y_pred_rf, alpha=0.3, s=20, color='#2196F3')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect prediction')
axes[0].set_xlabel('Actual log(Salary)', fontsize=11)
axes[0].set_ylabel('Predicted log(Salary)', fontsize=11)
axes[0].set_title(f'RF: Predicted vs Actual (log scale)\nR² = {rf_r2:.4f}', fontsize=12, fontweight='bold')
axes[0].legend()

residuals = y_test - y_pred_rf
axes[1].scatter(y_pred_rf, residuals, alpha=0.3, s=20, color='#4CAF50')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
axes[1].set_xlabel('Predicted log(Salary)', fontsize=11)
axes[1].set_ylabel('Residuals', fontsize=11)
axes[1].set_title('RF: Residual Plot', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '07_rf_predictions.png'), dpi=150, bbox_inches='tight')
plt.close()

# =============================================
# MODEL 2: XGBoost Regressor
# =============================================
print("\n--- XGBoost Regressor ---")
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbosity=0
    ))
])

xgb_pipeline.fit(X_train, y_train)
y_pred_xgb = xgb_pipeline.predict(X_test)

xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
xgb_r2 = r2_score(y_test, y_pred_xgb)

xgb_pred_actual = np.exp(y_pred_xgb)
xgb_mae_actual = mean_absolute_error(y_test_actual, xgb_pred_actual)
xgb_mape = np.mean(np.abs((y_test_actual - xgb_pred_actual) / y_test_actual)) * 100

print(f"  R²: {xgb_r2:.4f}, RMSE: {xgb_rmse:.4f}, MAE: {xgb_mae:.4f}")
print(f"  MAE (₹): ₹{xgb_mae_actual:,.0f}, MAPE: {xgb_mape:.2f}%")

results['xgb_r2'] = round(xgb_r2, 4)
results['xgb_rmse'] = round(xgb_rmse, 4)
results['xgb_mae_log'] = round(xgb_mae, 4)
results['xgb_mae_actual'] = round(xgb_mae_actual, 2)
results['xgb_mape'] = round(xgb_mape, 2)

# XGBoost Cross-Validation
xgb_cv = cross_val_score(xgb_pipeline, X, y, cv=5, scoring='r2', n_jobs=-1)
results['xgb_cv_mean'] = round(xgb_cv.mean(), 4)
results['xgb_cv_std'] = round(xgb_cv.std(), 4)
print(f"  5-Fold CV R²: {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")

# =============================================
# MODEL 3: ElasticNet (Linear Baseline)
# =============================================
print("\n--- ElasticNet (Linear Baseline) ---")
en_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42))
])

en_pipeline.fit(X_train, y_train)
y_pred_en = en_pipeline.predict(X_test)

en_r2 = r2_score(y_test, y_pred_en)
en_rmse = np.sqrt(mean_squared_error(y_test, y_pred_en))
en_mae = mean_absolute_error(y_test, y_pred_en)

en_pred_actual = np.exp(y_pred_en)
en_mae_actual = mean_absolute_error(y_test_actual, en_pred_actual)
en_mape = np.mean(np.abs((y_test_actual - en_pred_actual) / y_test_actual)) * 100

print(f"  R²: {en_r2:.4f}, RMSE: {en_rmse:.4f}")
print(f"  MAE (₹): ₹{en_mae_actual:,.0f}, MAPE: {en_mape:.2f}%")

results['en_r2'] = round(en_r2, 4)
results['en_rmse'] = round(en_rmse, 4)
results['en_mae_actual'] = round(en_mae_actual, 2)
results['en_mape'] = round(en_mape, 2)

# ElasticNet CV
en_cv = cross_val_score(en_pipeline, X, y, cv=5, scoring='r2', n_jobs=-1)
results['en_cv_mean'] = round(en_cv.mean(), 4)
results['en_cv_std'] = round(en_cv.std(), 4)
print(f"  5-Fold CV R²: {en_cv.mean():.4f} ± {en_cv.std():.4f}")

# =============================================
# MODEL 4: Linear Regression (Baseline reference)
# =============================================
print("\n--- Linear Regression (Baseline) ---")
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)

lr_r2 = r2_score(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))

lr_pred_actual = np.exp(y_pred_lr)
lr_mae_actual = mean_absolute_error(y_test_actual, lr_pred_actual)
lr_mape = np.mean(np.abs((y_test_actual - lr_pred_actual) / y_test_actual)) * 100

print(f"  R²: {lr_r2:.4f}, RMSE: {lr_rmse:.4f}")
print(f"  MAE (₹): ₹{lr_mae_actual:,.0f}, MAPE: {lr_mape:.2f}%")

results['lr_r2'] = round(lr_r2, 4)
results['lr_rmse'] = round(lr_rmse, 4)
results['lr_mae_actual'] = round(lr_mae_actual, 2)
results['lr_mape'] = round(lr_mape, 2)

lr_cv = cross_val_score(lr_pipeline, X, y, cv=5, scoring='r2', n_jobs=-1)
results['lr_cv_mean'] = round(lr_cv.mean(), 4)
results['lr_cv_std'] = round(lr_cv.std(), 4)
print(f"  5-Fold CV R²: {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")

# VISUALIZATION 8: Model Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

models = ['Linear\nRegression', 'ElasticNet', 'Random\nForest', 'XGBoost']
r2_vals = [lr_r2, en_r2, rf_r2, xgb_r2]
rmse_vals = [lr_rmse, en_rmse, rf_rmse, xgb_rmse]
cv_means = [lr_cv.mean(), en_cv.mean(), cv_scores.mean(), xgb_cv.mean()]
cv_stds = [lr_cv.std(), en_cv.std(), cv_scores.std(), xgb_cv.std()]

colors_models = ['#9E9E9E', '#FF9800', '#2196F3', '#4CAF50']

bars = axes[0].bar(models, r2_vals, color=colors_models, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, r2_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005, f'{val:.4f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
axes[0].set_ylabel('R² Score', fontsize=12)
axes[0].set_title('Test Set R² Score', fontsize=13, fontweight='bold')
axes[0].set_ylim(0, max(r2_vals) * 1.15)

axes[1].bar(models, cv_means, yerr=cv_stds, color=colors_models, edgecolor='white', 
            linewidth=1.5, capsize=5, error_kw={'linewidth': 2})
for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
    axes[1].text(i, m + s + 0.005, f'{m:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
axes[1].set_ylabel('R² Score', fontsize=12)
axes[1].set_title('5-Fold Cross-Validation R² (mean ± std)', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, max(cv_means) * 1.15)

plt.suptitle('Model Comparison - Predicting log(Starting Salary)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '08_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# =============================================
# VISUALIZATION 9: Salary by Degree / Specialization
# =============================================
if 'Degree' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    degree_order = df.groupby('Degree')['Starting_Salary'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='Degree', y='Starting_Salary', order=degree_order, ax=axes[0], palette='Set2')
    axes[0].set_title('Salary by Degree Type', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_ylabel('Starting Salary (₹)')
    
    if 'MajorSpecialization' in df.columns:
        top_specs = df['MajorSpecialization'].value_counts().head(10).index
        df_top_spec = df[df['MajorSpecialization'].isin(top_specs)]
        spec_order = df_top_spec.groupby('MajorSpecialization')['Starting_Salary'].median().sort_values(ascending=False).index
        sns.boxplot(data=df_top_spec, x='MajorSpecialization', y='Starting_Salary', order=spec_order, ax=axes[1], palette='Set3')
        axes[1].set_title('Salary by Top 10 Specializations', fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_ylabel('Starting Salary (₹)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, '09_salary_by_category.png'), dpi=150, bbox_inches='tight')
    plt.close()

# =============================================
# VISUALIZATION 10: College Tier Analysis
# =============================================
if 'CollegeTier' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    tier_stats = df.groupby('CollegeTier')['Starting_Salary'].agg(['mean', 'median', 'count'])
    axes[0].bar(tier_stats.index.astype(str), tier_stats['median'], color=['#4CAF50', '#FF9800', '#f44336'], edgecolor='white')
    axes[0].set_title('Median Salary by College Tier', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Median Salary (₹)')
    for i, (idx, row) in enumerate(tier_stats.iterrows()):
        axes[0].text(i, row['median'] + 2000, f'₹{row["median"]:,.0f}\n(n={int(row["count"])})', ha='center', fontsize=9)
    
    axes[1].bar(tier_stats.index.astype(str), tier_stats['count'], color=['#4CAF50', '#FF9800', '#f44336'], edgecolor='white')
    axes[1].set_title('Graduate Count by Tier', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, '10_college_tier.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Welch's t-test: Tier 1 vs Tier 2
    if len(tier_stats) >= 2:
        tier1 = df[df['CollegeTier'] == 1]['Starting_Salary'].dropna()
        tier2 = df[df['CollegeTier'] == 2]['Starting_Salary'].dropna()
        t_tier, p_tier = stats.ttest_ind(tier1, tier2, equal_var=False)
        d_cohen = (tier1.mean() - tier2.mean()) / np.sqrt((tier1.std()**2 + tier2.std()**2) / 2)
        results['tier_ttest_t'] = round(t_tier, 3)
        results['tier_ttest_p'] = f"{p_tier:.4e}"
        results['tier_cohen_d'] = round(d_cohen, 4)
        print(f"\nTier 1 vs Tier 2: t={t_tier:.3f}, p={p_tier:.4e}, Cohen's d={d_cohen:.4f}")

# =============================================
# SAVE INTERMEDIATE RESULTS (before Part B)
# =============================================
def convert_types(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(i) for i in obj]
    return obj

results_converted = convert_types(results)
with open('analysis_results.json', 'w') as f:
    json.dump(results_converted, f, indent=2, default=str)
print("Intermediate results saved to analysis_results.json")

# =============================================
# PART B: JOB MARKET ANALYSIS (INDEED INDIA)
# =============================================
print("\n" + "=" * 60)
print("PART B: Indeed India Job Market Analysis")
print("=" * 60)

# Load LDJSON
jobs = []
try:
    with open(LDJSON_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    jobs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
except FileNotFoundError:
    # Try alternate path
    alt_path = os.path.join(DATA_DIR, "marketing_sample_for_indeed_co_in-indeed_co_in_job_data__20211001_20211231__30k_data.ldjson")
    if os.path.isfile(alt_path):
        with open(alt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        jobs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

jobs_df = pd.DataFrame(jobs)
print(f"Job postings loaded: {len(jobs_df)}")
print(f"Columns: {list(jobs_df.columns)}")
results['n_jobs'] = len(jobs_df)

# Skills extraction from job descriptions
skills_keywords = [
    'Communication', 'Leadership', 'SQL', 'Excel', 'Java', 'Python', 'JavaScript',
    'HTML', 'CSS', 'React', 'Node', 'AWS', 'Docker', 'Kubernetes', 'Tableau',
    'Power BI', 'Machine Learning', 'Data Analysis', 'Project Management',
    'Agile', 'Scrum', 'Git', 'Linux', 'C++', 'C#', 'PHP', 'Ruby',
    'MongoDB', 'PostgreSQL', 'MySQL', 'Azure', 'GCP', 'Spark',
    'TensorFlow', 'Deep Learning', 'NLP', 'R', 'Scala', 'Hadoop',
    'Salesforce', 'SAP', 'Angular', 'Vue', 'TypeScript', 'REST API'
]

# Find the description column
desc_col = None
for col in jobs_df.columns:
    if 'description' in col.lower() or 'desc' in col.lower():
        desc_col = col
        break

if desc_col is None:
    # Check for job_description or similar
    for col in jobs_df.columns:
        if jobs_df[col].dtype == 'object' and jobs_df[col].str.len().mean() > 100:
            desc_col = col
            break

if desc_col:
    print(f"Using description column: {desc_col}")
    jobs_df[desc_col] = jobs_df[desc_col].fillna('')
    
    skill_counts = {}
    for skill in skills_keywords:
        pattern = r'\b' + skill.replace('+', r'\+').replace('#', r'\#') + r'\b'
        count = jobs_df[desc_col].str.contains(pattern, case=False, na=False).sum()
        if count > 0:
            skill_counts[skill] = count
    
    skill_series = pd.Series(skill_counts).sort_values(ascending=False)
    print(f"\nTop 20 Skills in Job Postings:")
    print(skill_series.head(20))
    results['top_skills'] = skill_series.head(20).to_dict()
    
    # VISUALIZATION 11: Top Skills Bar Chart
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        top_skills = skill_series.head(20)
        y_pos = list(range(len(top_skills)))
        y_pos.reverse()
        ax.barh(y_pos, top_skills.values, color='#2196F3', edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_skills.index.tolist(), fontsize=10)
        ax.set_xlabel('Number of Job Postings', fontsize=12)
        ax.set_title('Top 20 In-Demand Skills - Indeed India (Oct-Dec 2021)', fontsize=13, fontweight='bold')
        for i, count_val in enumerate(top_skills.values):
            ax.text(count_val + 50, y_pos[i], str(count_val), va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, '11_top_skills.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved 11_top_skills.png")
    except Exception as e:
        print(f"  WARNING: Could not save skill chart: {e}")
        plt.close('all')
    
    # VISUALIZATION 12: Skill Correlation Heatmap
    try:
        skill_matrix = pd.DataFrame()
        for skill in top_skills.index[:15]:
            pattern = r'\b' + skill.replace('+', r'\+').replace('#', r'\#') + r'\b'
            skill_matrix[skill] = jobs_df[desc_col].str.contains(pattern, case=False, na=False).astype(int)
        
        skill_corr = skill_matrix.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(skill_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax, 
                    square=True, linewidths=0.5, vmin=-0.3, vmax=0.7)
        ax.set_title('Skill Co-occurrence Correlation (Top 15 Skills)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, '12_skill_correlation.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved 12_skill_correlation.png")
    except Exception as e:
        print(f"  WARNING: Could not save skill correlation: {e}")
        plt.close('all')

    # Chi-Square: Skills vs Job Type
    job_type_col = None
    for col in jobs_df.columns:
        if 'type' in col.lower() or 'job_type' in col.lower():
            job_type_col = col
            break
    
    if job_type_col:
        print(f"\nUsing job type column: {job_type_col}")
        chi2_skills_results = []
        for skill in top_skills.index[:10]:
            pattern = r'\b' + skill.replace('+', r'\+').replace('#', r'\#') + r'\b'
            has_skill = jobs_df[desc_col].str.contains(pattern, case=False, na=False)
            contingency = pd.crosstab(has_skill, jobs_df[job_type_col])
            if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                chi2, p, dof, expected = stats.chi2_contingency(contingency)
                chi2_skills_results.append({
                    'skill': skill,
                    'chi2': round(chi2, 1),
                    'p_value': f"{p:.4e}",
                    'significant': p < 0.05
                })
                print(f"  {skill}: χ²={chi2:.1f}, p={p:.4e}")
        results['chi2_skills_jobtype'] = chi2_skills_results
else:
    print("WARNING: Could not find description column in job data")
    print(f"Available columns: {list(jobs_df.columns)}")

# =============================================
# VISUALIZATION 13: Gap Analysis Visualization
# =============================================
# Map graduate skills to job demands
graduate_skills = {
    'Programming': df['ProgrammingSkills'].mean() if 'ProgrammingSkills' in df.columns else 0,
    'Quantitative': df['QuantitativeAptitudeScore'].mean() if 'QuantitativeAptitudeScore' in df.columns else 0,
    'Logical Reasoning': df['LogicalReasoningScore'].mean() if 'LogicalReasoningScore' in df.columns else 0,
    'English': df['EnglishScore'].mean() if 'EnglishScore' in df.columns else 0,
    'College GPA': df['collegeGPA'].mean() if 'collegeGPA' in df.columns else 0,
}
results['graduate_skill_means'] = {k: round(v, 2) for k, v in graduate_skills.items()}

# =============================================
# ADDITIONAL: Welch's t-test for Gender salary gap
# =============================================
if 'Gender' in df.columns:
    males = df[df['Gender'] == 'm']['Starting_Salary'].dropna()
    females = df[df['Gender'] == 'f']['Starting_Salary'].dropna()
    if len(males) > 10 and len(females) > 10:
        t_gender, p_gender = stats.ttest_ind(males, females, equal_var=False)
        d_gender = (males.mean() - females.mean()) / np.sqrt((males.std()**2 + females.std()**2) / 2)
        results['gender_ttest_t'] = round(t_gender, 3)
        results['gender_ttest_p'] = f"{p_gender:.4e}"
        results['gender_cohen_d'] = round(d_gender, 4)
        results['gender_male_mean'] = round(males.mean(), 2)
        results['gender_female_mean'] = round(females.mean(), 2)
        print(f"\nGender salary gap: t={t_gender:.3f}, p={p_gender:.4e}, d={d_gender:.4f}")
        print(f"  Male mean: ₹{males.mean():,.0f}, Female mean: ₹{females.mean():,.0f}")

# =============================================
# SAVE ALL RESULTS
# =============================================
# Convert any numpy types for JSON serialization
def convert_types(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(i) for i in obj]
    return obj

results = convert_types(results)

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"Results saved to: analysis_results.json")
print(f"Visualizations saved to: {IMG_DIR}/")
print(f"Total images generated: {len(os.listdir(IMG_DIR))}")

# Print summary for HTML generation
print("\n--- SUMMARY FOR HTML ---")
print(f"Dataset: {results['n_graduates']} graduates, {results.get('n_jobs', 'N/A')} job postings")
print(f"GPA-Salary Pearson r: {results['pearson_gpa_salary_r']}")
print(f"RF R² (log salary): {results['rf_r2']}, CV: {results['rf_cv_mean']} ± {results['rf_cv_std']}")
print(f"XGBoost R²: {results['xgb_r2']}, CV: {results['xgb_cv_mean']} ± {results['xgb_cv_std']}")
print(f"ElasticNet R²: {results['en_r2']}, CV: {results['en_cv_mean']} ± {results['en_cv_std']}")
print(f"Linear Reg R²: {results['lr_r2']}, CV: {results['lr_cv_mean']} ± {results['lr_cv_std']}")
print(f"Best model: {'XGBoost' if xgb_r2 > rf_r2 else 'Random Forest'} (R²={max(xgb_r2, rf_r2):.4f})")
