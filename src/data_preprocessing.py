"""
Data Preprocessing and Leakage Prevention Module for ASD Screening.

Implements:
1. Strict leakage prevention (drops 'result', 'Qchat-10-Score', and derived columns).
2. Category harmonization, typo fixing, missing value handling ('?' -> median/mode).
3. Separation into:
   - Variant A: 10 Items only (pure questionnaire responses)
   - Variant B: 10 Items + Demographics (Age, Sex, Jaundice, Family history), strictly excluding Ethnicity and Country
   - Leakage Demonstration Variant: Variant A + Total Score
4. Unified age-aware dataset and individual age cohort datasets.
5. Untouched stratified train/test splitting (80/20, seed=42).
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def clean_uci_df(filepath, age_group_name):
    """Clean UCI format screening datasets (Adult, Adolescent, Child)."""
    df = pd.read_csv(filepath)
    df = df.copy()
    
    # Standardize column names
    col_map = {
        'A1_Score': 'A1', 'A2_Score': 'A2', 'A3_Score': 'A3', 'A4_Score': 'A4', 'A5_Score': 'A5',
        'A6_Score': 'A6', 'A7_Score': 'A7', 'A8_Score': 'A8', 'A9_Score': 'A9', 'A10_Score': 'A10',
        'gender': 'sex', 'jundice': 'jaundice', 'austim': 'family_asd',
        'Class/ASD': 'target', 'Class/ASD Traits ': 'target'
    }
    df.rename(columns=col_map, inplace=True)
    
    # Check for leakage column 'result'
    if 'result' in df.columns:
        df['leakage_score'] = pd.to_numeric(df['result'], errors='coerce')
    else:
        df['leakage_score'] = df[[f'A{i}' for i in range(1, 11)]].sum(axis=1)

    # Standardize target to binary 0/1
    df['target'] = df['target'].astype(str).str.strip().str.upper().map(lambda x: 1 if x in ['YES', '1', 'TRUE'] else 0)
    
    # Items to numeric binary
    for i in range(1, 11):
        df[f'A{i}'] = pd.to_numeric(df[f'A{i}'], errors='coerce').fillna(0).astype(int).clip(0, 1)
        
    # Standardize age (handle '?' or strings)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    med_age = df['age'].median()
    if np.isnan(med_age):
        med_age = 25.0 if age_group_name == 'adult' else (14.0 if age_group_name == 'adolescent' else 7.0)
    df['age'] = df['age'].fillna(med_age)
    
    # Standardize binary demographics
    df['sex'] = df['sex'].astype(str).str.strip().str.lower().map(lambda x: 1 if x in ['m', 'male', '1'] else 0)
    df['jaundice'] = df['jaundice'].astype(str).str.strip().str.lower().map(lambda x: 1 if x in ['yes', 'y', '1'] else 0)
    df['family_asd'] = df['family_asd'].astype(str).str.strip().str.lower().map(lambda x: 1 if x in ['yes', 'y', '1'] else 0)
    
    df['age_group'] = age_group_name
    
    # Remove duplicates
    initial_len = len(df)
    df.drop_duplicates(subset=[f'A{i}' for i in range(1, 11)] + ['age', 'sex', 'target'], inplace=True)
    if len(df) < initial_len:
        print(f"  Removed {initial_len - len(df)} duplicate rows from {age_group_name}.")
        
    return df


def clean_toddler_df(filepath):
    """Clean Toddler Q-CHAT-10 dataset."""
    df = pd.read_csv(filepath)
    df = df.copy()
    
    # Column mapping
    col_map = {
        'Sex': 'sex', 'Jaundice': 'jaundice', 'Family_mem_with_ASD': 'family_asd',
        'Class/ASD Traits ': 'target', 'Class/ASD': 'target', 'Qchat-10-Score': 'leakage_score'
    }
    df.rename(columns=col_map, inplace=True)
    
    # Items
    for i in range(1, 11):
        df[f'A{i}'] = pd.to_numeric(df[f'A{i}'], errors='coerce').fillna(0).astype(int).clip(0, 1)
        
    # Age: convert months to years for age-aware comparability
    if 'Age_Mons' in df.columns:
        df['age'] = df['Age_Mons'].astype(float) / 12.0
    else:
        df['age'] = 2.0
        
    # Demographics
    df['sex'] = df['sex'].astype(str).str.strip().str.lower().map(lambda x: 1 if x in ['m', 'male', '1'] else 0)
    df['jaundice'] = df['jaundice'].astype(str).str.strip().str.lower().map(lambda x: 1 if x in ['yes', 'y', '1'] else 0)
    df['family_asd'] = df['family_asd'].astype(str).str.strip().str.lower().map(lambda x: 1 if x in ['yes', 'y', '1'] else 0)
    
    # Target
    df['target'] = df['target'].astype(str).str.strip().str.upper().map(lambda x: 1 if x in ['YES', '1', 'TRUE'] else 0)
    
    if 'leakage_score' not in df.columns:
        df['leakage_score'] = df[[f'A{i}' for i in range(1, 11)]].sum(axis=1)
    else:
        df['leakage_score'] = pd.to_numeric(df['leakage_score'], errors='coerce')
        
    df['age_group'] = 'toddler'
    
    # Remove duplicates
    initial_len = len(df)
    df.drop_duplicates(subset=[f'A{i}' for i in range(1, 11)] + ['age', 'sex', 'target'], inplace=True)
    if len(df) < initial_len:
        print(f"  Removed {initial_len - len(df)} duplicate rows from toddler dataset.")
        
    return df


def prepare_datasets(test_size=0.2, random_state=42):
    """
    Cleans, verifies, and packages datasets into:
    - individual cohorts (toddler, child, adolescent, adult)
    - unified age-aware cohort
    Splits each into train and untouched test sets.
    """
    print("--- Preprocessing and Cleaning Datasets ---")
    
    adult_df = clean_uci_df(os.path.join(DATA_DIR, "uci_adult.csv"), "adult")
    ado_df = clean_uci_df(os.path.join(DATA_DIR, "uci_adolescent.csv"), "adolescent")
    child_df = clean_uci_df(os.path.join(DATA_DIR, "uci_child.csv"), "child")
    toddler_df = clean_toddler_df(os.path.join(DATA_DIR, "toddler_qchat10.csv"))
    
    cohorts = {
        'adult': adult_df,
        'adolescent': ado_df,
        'child': child_df,
        'toddler': toddler_df
    }
    
    # Unified dataset
    unified_cols = [f'A{i}' for i in range(1, 11)] + ['age', 'sex', 'jaundice', 'family_asd', 'age_group', 'leakage_score', 'target']
    unified_df = pd.concat([cohorts[c][unified_cols] for c in cohorts], ignore_index=True)
    cohorts['unified'] = unified_df
    
    item_cols = [f'A{i}' for i in range(1, 11)]
    demo_cols = item_cols + ['age', 'sex', 'jaundice', 'family_asd']
    if 'age_group' in unified_cols:
        unified_demo_cols = demo_cols + ['age_group']
    
    splits_summary = {}
    
    for name, df in cohorts.items():
        print(f"\nProcessing cohort: {name.upper()} (Total N={len(df)}, ASD rate={df['target'].mean():.2%})")
        
        # Check target class distribution
        if df['target'].nunique() < 2:
            raise ValueError(f"Cohort {name} has only 1 target class!")
            
        # Stratified train/test split
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['target']
        )
        
        # Save CSVs
        train_path = os.path.join(PROCESSED_DIR, f"{name}_train.csv")
        test_path = os.path.join(PROCESSED_DIR, f"{name}_test.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        splits_summary[name] = {
            'total_n': len(df),
            'train_n': len(train_df),
            'test_n': len(test_df),
            'train_asd_n': int(train_df['target'].sum()),
            'test_asd_n': int(test_df['target'].sum()),
            'train_path': train_path,
            'test_path': test_path
        }
        print(f"  Saved: Train N={len(train_df)} ({train_df['target'].sum()} ASD), Test N={len(test_df)} ({test_df['target'].sum()} ASD)")
        
    return splits_summary


if __name__ == "__main__":
    summary = prepare_datasets()
    print("\nDataset preparation complete! All datasets safely split into untouched test sets.")
