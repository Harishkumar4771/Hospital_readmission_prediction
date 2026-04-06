import os
import json
import requests
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

app = Flask(__name__)

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

THRESHOLD = 0.42

# Individual model filenames for comparison feature
MODEL_FILES = {
    'lr':       'model_lr.pkl',
    'rf':       'model_rf.pkl',
    'xgb':      'model_xgb.pkl',
    'lgbm':     'model_lgbm.pkl',
    'ensemble': 'model.pkl',
}

MODEL_LABELS = {
    'lr':       'Logistic Regression',
    'rf':       'Random Forest',
    'xgb':      'XGBoost',
    'lgbm':     'LightGBM',
    'ensemble': 'Soft Voting Ensemble',
}

IQR_CAPS = {
    'age':              (25.0,   105.0),
    'time_in_hospital': (-4.0,   12.0),
    'n_lab_procedures': (-8.0,   96.0),
    'n_procedures':     (-3.0,   5.0),
    'n_medications':    (-2.5,   33.5),
    'n_outpatient':     (0.0,    0.0),
    'n_inpatient':      (-1.5,   2.5),
    'n_emergency':      (0.0,    0.0),
    'total_visits':     (-3.0,   5.0),
    'visit_severity':   (-4.5,   7.5),
    'meds_per_day':     (-2.94,  11.90),
}

CATEGORICAL_FIELDS = {
    'medical_specialty': ['Cardiology', 'Emergency/Trauma', 'Family/GeneralPractice',
                          'InternalMedicine', 'Missing', 'Other', 'Surgery'],
    'diag_1':            ['Circulatory', 'Diabetes', 'Digestive', 'Injury',
                          'Missing', 'Musculoskeletal', 'Other', 'Respiratory'],
    'diag_2':            ['Circulatory', 'Diabetes', 'Digestive', 'Injury',
                          'Missing', 'Musculoskeletal', 'Other', 'Respiratory'],
    'diag_3':            ['Circulatory', 'Diabetes', 'Digestive', 'Injury',
                          'Missing', 'Musculoskeletal', 'Other', 'Respiratory'],
    'glucose_test':      ['high', 'no', 'normal'],
    'A1Ctest':           ['high', 'no', 'normal'],
    'change':            ['no', 'yes'],
    'diabetes_med':      ['no', 'yes'],
}

AGE_MAP = {
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65,
    '[70-80)': 75, '[80-90)': 85, '[90-100)': 95
}

# ──────────────────────────────────────────────
# MODEL TRAINING
# ──────────────────────────────────────────────

def train_and_save():
    print("Training all models from scratch...")
    df = pd.read_csv('df_encoded.csv')
    X  = df.drop('readmitted', axis=1)
    y  = df['readmitted']

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr_model = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42)
    rf_model = RandomForestClassifier(
        n_estimators=250, max_depth=6, min_samples_split=10,
        min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1
    )
    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, gamma=1, min_child_weight=5,
        reg_alpha=0.5, reg_lambda=2, eval_metric='logloss', random_state=42
    )
    lgbm_model = LGBMClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=5,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=30, reg_alpha=0.5, reg_lambda=1.5,
        class_weight='balanced', random_state=42, verbose=-1
    )

    # Train individual models
    print("  Training Logistic Regression...")
    lr_model.fit(X_scaled, y)
    joblib.dump(lr_model, MODEL_FILES['lr'], compress=3)

    print("  Training Random Forest...")
    rf_model.fit(X_scaled, y)
    joblib.dump(rf_model, MODEL_FILES['rf'], compress=3)

    print("  Training XGBoost...")
    xgb_model.fit(X_scaled, y)
    joblib.dump(xgb_model, MODEL_FILES['xgb'], compress=3)

    print("  Training LightGBM...")
    lgbm_model.fit(X_scaled, y)
    joblib.dump(lgbm_model, MODEL_FILES['lgbm'], compress=3)

    # Train ensemble (reuses already-fitted estimators via fresh instances)
    print("  Training Soft Voting Ensemble...")
    lr2  = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42)
    rf2  = RandomForestClassifier(n_estimators=250, max_depth=6, min_samples_split=10,
                                   min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1)
    xgb2 = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, gamma=1, min_child_weight=5,
                          reg_alpha=0.5, reg_lambda=2, eval_metric='logloss', random_state=42)
    ensemble = VotingClassifier(
        estimators=[('lr', lr2), ('rf', rf2), ('xgb', xgb2)],
        voting='soft', weights=[1, 1, 2]
    )
    ensemble.fit(X_scaled, y)
    joblib.dump(ensemble, MODEL_FILES['ensemble'], compress=3)
    joblib.dump(scaler,   'scaler.pkl', compress=3)

    print(f"All models trained and saved. Threshold: {THRESHOLD}")
    return ensemble, scaler


def load_individual_models():
    """Load all individual models into a dict. Train if any are missing."""
    models = {}
    all_exist = all(os.path.exists(f) for f in MODEL_FILES.values())
    if not all_exist:
        print("Some individual models missing — retraining all...")
        train_and_save()
    for key, fname in MODEL_FILES.items():
        try:
            models[key] = joblib.load(fname)
        except Exception as e:
            print(f"Could not load {fname}: {e}")
    return models


def load_or_train():
    if os.path.exists('model.pkl') and os.path.exists('scaler.pkl'):
        try:
            m = joblib.load('model.pkl')
            s = joblib.load('scaler.pkl')
            if (isinstance(m, VotingClassifier) and
                    m.voting == 'soft' and list(m.weights) == [1, 1, 2]):
                print("Loaded correct soft-voting model.")
                return m, s
            else:
                print("Outdated model detected. Retraining...")
        except Exception as e:
            print(f"Model load error: {e}. Retraining...")
    return train_and_save()


# ──────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────

def apply_iqr_caps(data: dict) -> dict:
    capped = data.copy()
    for col, (lower, upper) in IQR_CAPS.items():
        if col in capped:
            capped[col] = float(np.clip(capped[col], lower, upper))
    return capped


def build_feature_vector(form_data: dict) -> pd.DataFrame:
    age_val          = AGE_MAP.get(form_data.get('age', '[60-70)'), 65)
    time_in_hospital = float(form_data.get('time_in_hospital', 1))
    n_lab_procedures = float(form_data.get('n_lab_procedures', 0))
    n_procedures     = float(form_data.get('n_procedures', 0))
    n_medications    = float(form_data.get('n_medications', 0))
    n_outpatient     = float(form_data.get('n_outpatient', 0))
    n_inpatient      = float(form_data.get('n_inpatient', 0))
    n_emergency      = float(form_data.get('n_emergency', 0))

    total_visits   = n_inpatient + n_outpatient + n_emergency
    visit_severity = 3 * n_inpatient + 2 * n_emergency + 1 * n_outpatient
    polypharmacy   = 1 if n_medications >= 5 else 0
    meds_per_day   = n_medications / time_in_hospital if time_in_hospital > 0 else 0
    long_stay      = 1 if time_in_hospital > 7 else 0

    base = {
        'age': age_val, 'time_in_hospital': time_in_hospital,
        'n_lab_procedures': n_lab_procedures, 'n_procedures': n_procedures,
        'n_medications': n_medications, 'n_outpatient': n_outpatient,
        'n_inpatient': n_inpatient, 'n_emergency': n_emergency,
        'total_visits': total_visits, 'visit_severity': visit_severity,
        'polypharmacy': polypharmacy, 'meds_per_day': meds_per_day,
        'long_stay': long_stay,
    }
    base     = apply_iqr_caps(base)
    input_df = pd.DataFrame([base])

    for field, categories in CATEGORICAL_FIELDS.items():
        user_choice = form_data.get(field, '')
        for cat in categories:
            input_df[f'{field}_{cat}'] = 1.0 if user_choice == cat else 0.0

    df_train     = pd.read_csv('df_encoded.csv')
    feature_cols = [c for c in df_train.columns if c != 'readmitted']
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0.0
    return input_df[feature_cols]


def run_prediction(form_data: dict) -> dict:
    input_df   = build_feature_vector(form_data)
    scaled     = scaler.transform(input_df)
    proba      = model.predict_proba(scaled)[0][1]
    prediction = 1 if proba >= THRESHOLD else 0

    if proba < 0.35:
        risk_level = 'Low'
    elif proba < 0.60:
        risk_level = 'Moderate'
    else:
        risk_level = 'High'

    # Convert all numeric fields to float so Jinja template comparisons work
    _f = lambda k, d=0: float(form_data.get(k, d) or d)
    key_factors = {
        'age':              form_data.get('age', 'Unknown'),
        'time_in_hospital': _f('time_in_hospital'),
        'n_medications':    _f('n_medications'),
        'n_inpatient':      _f('n_inpatient'),
        'n_outpatient':     _f('n_outpatient'),
        'n_emergency':      _f('n_emergency'),
        'n_lab_procedures': _f('n_lab_procedures'),
        'diag_1':           form_data.get('diag_1', 'Unknown'),
        'diag_2':           form_data.get('diag_2', 'Unknown'),
        'glucose_test':     form_data.get('glucose_test', 'no'),
        'A1Ctest':          form_data.get('A1Ctest', 'no'),
        'change':           form_data.get('change', 'no'),
        'diabetes_med':     form_data.get('diabetes_med', 'no'),
        'polypharmacy':     1 if _f('n_medications') >= 5 else 0,
        'medical_specialty': form_data.get('medical_specialty', 'Unknown'),
    }

    return {
        'prediction':  prediction,
        'probability': round(float(proba), 4),
        'risk_level':  risk_level,
        'key_factors': key_factors,
        'form_data':   form_data,
    }


# ──────────────────────────────────────────────
# AI SUMMARY HELPER
# ──────────────────────────────────────────────

def generate_ai_summary(prediction, probability, risk_level, key_factors):
    """
    Calls Anthropic Claude API to generate a patient-specific clinical summary.
    Requires ANTHROPIC_API_KEY environment variable.
    Falls back to a rule-based summary if API key is not set.
    """
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')

    polypharmacy_txt = 'present (≥5 medications)' if key_factors.get('polypharmacy') else 'not flagged'
    glucose_txt      = key_factors.get('glucose_test', 'not performed')
    a1c_txt          = key_factors.get('A1Ctest', 'not performed')
    change_txt       = 'yes — medication regimen was modified during stay' if key_factors.get('change') == 'yes' else 'no'
    diabetes_txt     = 'yes' if key_factors.get('diabetes_med') == 'yes' else 'no'

    prompt = f"""You are a clinical decision support system. Generate a concise 3-paragraph patient readmission risk summary.

Patient Data:
- Readmission prediction: {'WILL be readmitted' if prediction == 1 else 'Will NOT be readmitted'}
- Readmission probability: {round(probability * 100, 1)}%
- Risk classification: {risk_level}
- Age group: {key_factors.get('age', 'Unknown')}
- Hospital stay duration: {key_factors.get('time_in_hospital', 'Unknown')} days
- Medications: {key_factors.get('n_medications', 'Unknown')}
- Polypharmacy: {polypharmacy_txt}
- Prior inpatient admissions (past year): {key_factors.get('n_inpatient', 'Unknown')}
- Prior emergency visits (past year): {key_factors.get('n_emergency', 'Unknown')}
- Lab procedures performed: {key_factors.get('n_lab_procedures', 'Unknown')}
- Primary diagnosis category: {key_factors.get('diag_1', 'Unknown')}
- Secondary diagnosis: {key_factors.get('diag_2', 'Unknown')}
- Glucose test result: {glucose_txt}
- A1C test result: {a1c_txt}
- Diabetes medication prescribed: {diabetes_txt}
- Medication change during stay: {change_txt}
- Medical specialty: {key_factors.get('medical_specialty', 'Unknown')}

Write exactly 3 short paragraphs:
Paragraph 1: Overall risk assessment — what the {round(probability * 100, 1)}% probability means clinically and what risk level {risk_level} implies for this patient's post-discharge trajectory.
Paragraph 2: Key contributing clinical factors identified from the data — explain which specific values are driving risk up or down and why.
Paragraph 3: Specific recommended follow-up actions for the care team based on this patient's profile.

Be clinical, specific, and actionable. Do not use bullet points. Write in full sentences. Do not mention AI or machine learning. Write as a clinical decision support tool would."""

    if api_key:
        try:
            headers = {
                'Content-Type':         'application/json',
                'x-api-key':            api_key,
                'anthropic-version':    '2023-06-01',
            }
            payload = {
                'model':      'claude-haiku-4-5-20251001',
                'max_tokens': 500,
                'messages':   [{'role': 'user', 'content': prompt}]
            }
            resp = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers=headers,
                json=payload,
                timeout=15
            )
            if resp.status_code == 200:
                return resp.json()['content'][0]['text']
        except Exception as e:
            app.logger.error(f'Anthropic API error: {e}')

    # ── Fallback: rule-based summary ──────────────────────
    prob_pct   = round(probability * 100, 1)
    n_inpat    = int(key_factors.get('n_inpatient', 0))
    n_meds     = int(key_factors.get('n_medications', 0))
    n_days     = int(key_factors.get('time_in_hospital', 0))
    diag       = key_factors.get('diag_1', 'Unknown')
    poly_flag  = key_factors.get('polypharmacy', 0)

    risk_desc = {
        'High':     f'The model assigns a {prob_pct}% readmission probability, placing this patient in the high-risk category. Patients at this level require active care coordination before discharge and structured follow-up within 48–72 hours.',
        'Moderate': f'The model assigns a {prob_pct}% readmission probability, indicating moderate readmission risk. This patient warrants closer post-discharge monitoring and a follow-up appointment within 7 days.',
        'Low':      f'The model assigns a {prob_pct}% readmission probability, indicating low readmission risk. Standard discharge protocols are likely sufficient, though routine follow-up remains advisable.',
    }.get(risk_level, f'Readmission probability: {prob_pct}%.')

    factor_parts = []
    if n_inpat >= 2:
        factor_parts.append(f'{n_inpat} prior inpatient admissions in the past year — the strongest predictor of future readmission in this dataset')
    if poly_flag:
        factor_parts.append(f'polypharmacy ({n_meds} medications) indicating clinical complexity')
    if n_days > 7:
        factor_parts.append(f'an extended hospital stay of {n_days} days suggesting significant illness severity')
    if key_factors.get('A1Ctest') == 'high' or key_factors.get('glucose_test') == 'high':
        factor_parts.append('elevated glycaemic markers (high A1C or glucose) indicating suboptimal diabetes control')
    if not factor_parts:
        factor_parts.append(f'a primary diagnosis of {diag} and {n_days} days of hospitalisation')

    factors_txt = f"The primary contributors to this risk assessment include {', and '.join(factor_parts)}."

    action_map = {
        'High':     'The care team should arrange a discharge planning conference, schedule a follow-up appointment within 48 hours, review and reconcile the medication list before discharge, and consider referral to a community health worker or care navigator.',
        'Moderate': 'A follow-up appointment within 7 days is recommended. The team should ensure the patient has clear written discharge instructions, understands medication changes, and has a confirmed primary care contact.',
        'Low':      'Standard discharge procedures are appropriate. Ensure the patient has a scheduled follow-up within 30 days and understands the signs and symptoms warranting an earlier return to care.',
    }
    actions_txt = action_map.get(risk_level, 'Follow standard discharge protocols and schedule routine follow-up.')

    return f"{risk_desc}\n\n{factors_txt}\n\n{actions_txt}"


# ──────────────────────────────────────────────
# APP STARTUP
# ──────────────────────────────────────────────

model, scaler = load_or_train()
all_models    = load_individual_models()  # dict: {lr, rf, xgb, lgbm, ensemble}


# ──────────────────────────────────────────────
# PAGE ROUTES
# ──────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictor')
def predictor():
    return render_template('form.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how-it-works')
def how_it_works():
    return render_template('how-it-works.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/demo')
def demo():
    return render_template('demo.html')

@app.route('/ethics')
def ethics():
    return render_template('ethics.html')


# ──────────────────────────────────────────────
# PREDICTION ROUTES
# ──────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    """HTML form submission → renders result.html"""
    try:
        form_data = request.form.to_dict()
        result    = run_prediction(form_data)
        return render_template('result.html', **result)
    except Exception as e:
        app.logger.error(f'Prediction error: {e}')
        return render_template('error.html', error=str(e)), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    JSON API endpoint for programmatic access.

    POST /api/predict
    Content-Type: application/json

    Body fields: age, time_in_hospital, n_lab_procedures, n_procedures,
    n_medications, n_outpatient, n_inpatient, n_emergency, medical_specialty,
    diag_1, diag_2, diag_3, glucose_test, A1Ctest, change, diabetes_med
    """
    try:
        data   = request.get_json(force=True)
        result = run_prediction(data)
        return jsonify({
            'success':     True,
            'prediction':  result['prediction'],
            'probability': result['probability'],
            'risk_level':  result['risk_level'],
            'threshold':   THRESHOLD,
            'model_info': {
                'type':    'SoftVotingClassifier',
                'weights': [1, 1, 2],
                'models':  ['LogisticRegression', 'RandomForestClassifier', 'XGBClassifier'],
            }
        })
    except Exception as e:
        app.logger.error(f'API prediction error: {e}')
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai-summary', methods=['POST'])
def api_ai_summary():
    """
    Generates a patient-specific clinical summary using the Anthropic API.
    Requires ANTHROPIC_API_KEY environment variable.
    Gracefully falls back to a rule-based summary if key is not set.

    POST /api/ai-summary
    Body: { prediction, probability, risk_level, key_factors }
    """
    try:
        data        = request.get_json(force=True)
        prediction  = data.get('prediction', 0)
        probability = data.get('probability', 0.5)
        risk_level  = data.get('risk_level', 'Moderate')
        key_factors = data.get('key_factors', {})

        summary = generate_ai_summary(prediction, probability, risk_level, key_factors)
        return jsonify({'success': True, 'summary': summary})
    except Exception as e:
        app.logger.error(f'AI summary error: {e}')
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/model-info')
def api_model_info():
    """Returns model metadata — consumed by the Dashboard page."""
    return jsonify({
        'model_type':   'Soft Voting Ensemble',
        'components':   ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'weights':      [1, 1, 2],
        'threshold':    THRESHOLD,
        'dataset_size': 25000,
        'features':     54,
        'metrics': {
            'accuracy':       0.60,
            'recall_class_1': 0.75,
            'recall_class_0': 0.47,
            'roc_auc':        0.67,
        },
        'model_comparison': [
            {'model': 'Logistic Regression', 'accuracy': 0.6084, 'recall_1': 0.490, 'recall_0': 0.710},
            {'model': 'Random Forest',       'accuracy': 0.5978, 'recall_1': 0.502, 'recall_0': 0.680},
            {'model': 'XGBoost',             'accuracy': 0.6130, 'recall_1': 0.490, 'recall_0': 0.723},
            {'model': 'Voting Ensemble',     'accuracy': 0.5990, 'recall_1': 0.748, 'recall_0': 0.467},
            {'model': 'LightGBM',            'accuracy': 0.6080, 'recall_1': 0.684, 'recall_0': 0.541},
        ]
    })



@app.route('/compare')
def compare():
    return render_template('compare.html')


@app.route('/api/compare', methods=['POST'])
def api_compare():
    """
    Runs patient data through multiple models simultaneously.
    POST body: { form_data: {...}, models: ['lr','rf','xgb','lgbm','ensemble'] }
    Returns per-model predictions with probabilities.
    """
    try:
        data       = request.get_json(force=True)
        form_data  = data.get('form_data', {})
        model_keys = data.get('models', list(MODEL_FILES.keys()))

        input_df = build_feature_vector(form_data)
        scaled   = scaler.transform(input_df)

        results = {}
        for key in model_keys:
            if key not in all_models:
                continue
            m = all_models[key]
            try:
                proba      = float(m.predict_proba(scaled)[0][1])
                prediction = 1 if proba >= THRESHOLD else 0
                if proba < 0.35:
                    risk = 'Low'
                elif proba < 0.60:
                    risk = 'Moderate'
                else:
                    risk = 'High'
                results[key] = {
                    'label':      MODEL_LABELS[key],
                    'prediction': prediction,
                    'probability': round(proba, 4),
                    'risk_level': risk,
                }
            except Exception as e:
                results[key] = {'label': MODEL_LABELS[key], 'error': str(e)}

        return jsonify({'success': True, 'results': results, 'threshold': THRESHOLD})
    except Exception as e:
        app.logger.error(f'Compare error: {e}')
        return jsonify({'success': False, 'error': str(e)}), 500

# ──────────────────────────────────────────────
# PDF AUTOFILL ROUTE
# ──────────────────────────────────────────────

@app.route('/api/parse-pdf', methods=['POST'])
def api_parse_pdf():
    """
    Accepts a PDF file upload, extracts text using pypdf, then uses
    keyword/regex matching to autofill form fields. No API key required.
    """
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400

    pdf_file = request.files['file']
    if not pdf_file.filename.lower().endswith('.pdf'):
        return jsonify({'success': False, 'error': 'Only PDF files are supported'}), 400

    try:
        import io, re
        from pypdf import PdfReader

        pdf_bytes = pdf_file.read()
        reader    = PdfReader(io.BytesIO(pdf_bytes))
        text      = "\n".join(page.extract_text() or "" for page in reader.pages)
        text_low  = text.lower()

        if not text.strip():
            return jsonify({'success': False, 'error': 'Could not extract text — PDF may be scanned'}), 400

        extracted = {}

        # ── Age ────────────────────────────────────────────────────────
        age_val = None
        # Try: "Age: 72", "aged 72", "72 years", "72-year-old", "72 year old"
        for pattern in [
            r'\bage[:\s]+(\d{2,3})\s*(?:years?)?',
            r'\baged\s+(\d{2,3})\b',
            r'\b(\d{2,3})\s*-?\s*year[s]?\s*(?:old|of age)',
            r'\b(\d{2,3})\s*years?\s*old\b',
            r'(?:dob|date of birth)[^\d]*(\d{2,3})',
        ]:
            age_match = re.search(pattern, text_low)
            if age_match:
                age_val = int(age_match.group(1))
                break
        if age_val and 40 <= age_val < 100:
            for bracket in ['[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']:
                lo, hi = int(bracket[1:3]), int(bracket[4:6])
                if lo <= age_val < hi:
                    extracted['age'] = bracket
                    break

        # ── Length of stay ─────────────────────────────────────────────
        los = re.search(r'(?:length of stay|los|stayed?|admitted for)[^\d]*(\d+)\s*day', text_low)
        if not los:
            los = re.search(r'(\d+)\s*(?:-\s*)?day[s]?\s*(?:stay|admission|inpatient)', text_low)
        if los:
            extracted['time_in_hospital'] = int(los.group(1))

        # ── Lab procedures ─────────────────────────────────────────────
        lab = re.search(r'(?:lab(?:oratory)?\s*(?:procedures?|tests?|procedures?\s*performed))[^\d]*(\d+)', text_low)
        if lab:
            extracted['n_lab_procedures'] = int(lab.group(1))

        # ── Non-lab procedures ─────────────────────────────────────────
        proc = re.search(r'non-?lab(?:oratory)?\s*procedures?[^\d]*(\d+)', text_low)
        if proc:
            extracted['n_procedures'] = min(int(proc.group(1)), 5)
        else:
            extracted['n_procedures'] = 0

        # ── Medications ────────────────────────────────────────────────
        meds = re.search(r'(?:total\s*of\s*|medications?\s*(?:administered|count|:)\s*)(\d+)\s*medications?', text_low)
        if not meds:
            meds = re.search(r'(\d+)\s*medications?\s*(?:on discharge|administered|prescribed)', text_low)
        if meds:
            extracted['n_medications'] = int(meds.group(1))

        # ── Prior visits ───────────────────────────────────────────────
        inpat = re.search(r'inpatient\s*admissions?[^\d]*(\d+)', text_low)
        if inpat:
            extracted['n_inpatient'] = int(inpat.group(1))

        outpat = re.search(r'outpatient\s*visits?[^\d]*(\d+)', text_low)
        if outpat:
            extracted['n_outpatient'] = int(outpat.group(1))

        emerg = re.search(r'emergency\s*visits?[^\d]*(\d+)', text_low)
        if emerg:
            extracted['n_emergency'] = int(emerg.group(1))

        # ── Medical specialty ──────────────────────────────────────────
        specialty_map = {
            'cardiology':              'Cardiology',
            'emergency':               'Emergency/Trauma',
            'trauma':                  'Emergency/Trauma',
            'family':                  'Family/GeneralPractice',
            'general practice':        'Family/GeneralPractice',
            'internal medicine':       'InternalMedicine',
            'internalmedicine':        'InternalMedicine',
            'surgery':                 'Surgery',
        }
        for kw, val in specialty_map.items():
            if kw in text_low:
                extracted['medical_specialty'] = val
                break

        # ── Diagnoses ──────────────────────────────────────────────────
        diag_map = {
            'circulatory': 'Circulatory', 'heart': 'Circulatory', 'cardiac': 'Circulatory',
            'hypertens': 'Circulatory', 'vascular': 'Circulatory',
            'diabetes': 'Diabetes', 'diabetic': 'Diabetes',
            'digestive': 'Digestive', 'gastrointestinal': 'Digestive', 'liver': 'Digestive',
            'injury': 'Injury', 'trauma': 'Injury', 'fracture': 'Injury',
            'musculoskeletal': 'Musculoskeletal', 'arthritis': 'Musculoskeletal', 'bone': 'Musculoskeletal',
            'respiratory': 'Respiratory', 'pulmonary': 'Respiratory', 'lung': 'Respiratory', 'copd': 'Respiratory',
        }

        # Look for explicit diagnosis labels first
        diag_hits = []
        for label in ['primary diagnosis', 'secondary diagnosis', 'tertiary diagnosis',
                      'diag_1', 'diag_2', 'diag_3', 'additional diagnosis']:
            m = re.search(rf'{label}[:\s]+([^\n]+)', text_low)
            if m:
                snippet = m.group(1)
                for kw, val in diag_map.items():
                    if kw in snippet and val not in diag_hits:
                        diag_hits.append(val)
                        break

        # Fall back to scanning whole text
        if not diag_hits:
            for kw, val in diag_map.items():
                if kw in text_low and val not in diag_hits:
                    diag_hits.append(val)

        for i, key in enumerate(['diag_1', 'diag_2', 'diag_3']):
            if i < len(diag_hits):
                extracted[key] = diag_hits[i]

        # ── Glucose test ───────────────────────────────────────────────
        gluc = re.search(r'(?:fasting\s*blood\s*glucose|glucose)[^\n]*?(high|normal|low|elevated|\d+\s*mg)', text_low)
        if gluc:
            g = gluc.group(1)
            if 'high' in g or 'elevated' in g or re.search(r'1\d{2,}', g):
                extracted['glucose_test'] = 'high'
            elif 'normal' in g:
                extracted['glucose_test'] = 'normal'
        elif 'glucose' in text_low:
            extracted['glucose_test'] = 'no'

        # ── A1C test ───────────────────────────────────────────────────
        a1c = re.search(r'(?:hba1c|a1c|a1ctest)[^\n]*?(high|normal|\d+\.?\d*\s*%)', text_low)
        if a1c:
            val = a1c.group(1)
            if 'high' in val:
                extracted['A1Ctest'] = 'high'
            elif 'normal' in val:
                extracted['A1Ctest'] = 'normal'
            else:
                num = re.search(r'(\d+\.?\d*)', val)
                if num and float(num.group(1)) >= 6.5:
                    extracted['A1Ctest'] = 'high'
                elif num:
                    extracted['A1Ctest'] = 'normal'
        elif 'a1c' in text_low or 'hba1c' in text_low:
            extracted['A1Ctest'] = 'no'

        # ── Medication change ──────────────────────────────────────────
        if re.search(r'(?:medication|regimen|drug)\s*(?:was\s*)?(?:modified|changed|adjusted|revised|updated)', text_low):
            extracted['change'] = 'yes'
        else:
            extracted['change'] = 'no'

        # ── Diabetes medication ────────────────────────────────────────
        if re.search(r'(?:diabetes\s*medication|insulin|metformin|glipizide|glargine|oral\s*agent)', text_low):
            extracted['diabetes_med'] = 'yes'
        else:
            extracted['diabetes_med'] = 'no'

        return jsonify({'success': True, 'data': extracted})

    except Exception as e:
        app.logger.error(f'PDF parse error: {e}')
        return jsonify({'success': False, 'error': str(e)}), 500


# ──────────────────────────────────────────────
# ERROR HANDLERS
# ──────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error='Internal server error'), 500


# ──────────────────────────────────────────────
# STUDENT HEALTH CHECK — ROUTES & LOGIC
# ──────────────────────────────────────────────

HEALTH_CSV = 'student_health.csv'

HEALTH_REF = {
    # (low, high) — None means no lower bound check
    'hb_male':       (13.0,  17.0),
    'hb_female':     (12.0,  15.0),
    'rbc_male':      (4.5,   5.5),
    'rbc_female':    (3.8,   4.8),
    'platelets':     (150.0, 410.0),
}

# Categorical pass values
HEALTH_CAT_NORMAL = {
    'bile':     ['negative'],
    'glucose':  ['negative'],
    'crystals': ['absent'],
    # pus: absent or occasional = normal
    'pus':      ['absent', 'occasional'],
}


def parse_suhrc_pdf(pdf_bytes: bytes) -> dict:
    """
    Extract 9 health features from a Symbiosis CCL PDF.
    Handles variable page order (urine/CBC can appear in any order).
    Returns dict with keys: uhid, gender, hb, rbc, platelets,
                            bile, pus, crystals, glucose, error
    """
    import io, re
    from pypdf import PdfReader

    result = {
        'uhid': None, 'gender': None,
        'hb': None, 'rbc': None, 'platelets': None,
        'bile': None, 'pus': None, 'crystals': None, 'glucose': None,
        'error': None,
    }

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages_text = [p.extract_text() or '' for p in reader.pages]
        full_text  = '\n'.join(pages_text)

        if not full_text.strip():
            result['error'] = 'Could not extract text — PDF may be scanned or image-based.'
            return result

        # ── VALIDATE: Check if this is actually a medical lab report ──
        medical_keywords = ['hemoglobin', 'hb', 'rbc', 'glucose', 'platelets', 
                           'haematology', 'clinical pathology', 'urine', 'laboratory',
                           'bile', 'crystals', 'pus']
        text_lower = full_text.lower()
        found_keywords = sum(1 for kw in medical_keywords if kw in text_lower)
        
        if found_keywords < 3:
            result['error'] = 'Invalid PDF. Please upload a valid medical lab report with hemoglobin, glucose, or other lab values.'
            return result

        # ── Header fields (appear on every page, grab from full text) ──
        # Try to find UHID/ID field (may have different formats)
        uhid_m = re.search(r'(?:UHID|ID|Patient ID|Report ID)\s*:\s*(\S+)', full_text, re.IGNORECASE)
        if uhid_m:
            result['uhid'] = uhid_m.group(1).strip()
        else:
            # Generate a placeholder ID if not found
            result['uhid'] = 'TEST_' + str(hash(full_text[:100]))[:8]
        
        gender_m = re.search(r'(?:Gender|Sex)/Age\s*:\s*(Male|Female|M|F)', full_text, re.IGNORECASE)
        if gender_m:
            g = gender_m.group(1).upper()
            result['gender'] = 'Male' if g in ['M', 'MALE'] else 'Female'

        # ── Parse each page by section type ──
        for text in pages_text:
            tl = text.lower()

            # ── CBC / HAEMATOLOGY page ────────────────────────────────
            if 'haematology' in tl or 'hemoglobin' in tl:
                hb_m = re.search(
                    r'Hb\s*\(HEMOGLOBIN\)\s+([\d.]+)', text, re.IGNORECASE)
                if hb_m:
                    result['hb'] = float(hb_m.group(1))

                rbc_m = re.search(
                    r'RBC\s+Count\s+([\d.]+)', text, re.IGNORECASE)
                if rbc_m:
                    result['rbc'] = float(rbc_m.group(1))

                plt_m = re.search(
                    r'PLATELET\s+COUNT\s+([\d.]+)', text, re.IGNORECASE)
                if plt_m:
                    result['platelets'] = float(plt_m.group(1))

            # ── Urine / Clinical Pathology page ──────────────────────
            if 'clinical pathology' in tl or 'urine' in tl:
                bile_m = re.search(
                    r'BILE\s+PIGMENT\s+(\w+)', text, re.IGNORECASE)
                if bile_m:
                    result['bile'] = bile_m.group(1).capitalize()

                pus_m = re.search(
                    r'PUS\s+CELLS\s+(\w+)', text, re.IGNORECASE)
                if pus_m:
                    result['pus'] = pus_m.group(1).capitalize()

                crys_m = re.search(
                    r'CRYSTALS\s+(\w+)', text, re.IGNORECASE)
                if crys_m:
                    result['crystals'] = crys_m.group(1).capitalize()

                gluc_m = re.search(
                    r'GLUCOSE\s*:\s*(\w+)', text, re.IGNORECASE)
                if gluc_m:
                    result['glucose'] = gluc_m.group(1).capitalize()

    except Exception as e:
        result['error'] = str(e)

    return result


def run_health_rule_engine(data: dict) -> dict:
    """
    Apply clinical reference ranges to extracted data.
    Returns per-parameter flags and overall recommendation.
    """
    gender   = (data.get('gender') or 'Male').capitalize()
    abnormal = []
    flags    = {}

    def flag(key, value, low, high, unit='', label=None):
        lbl = label or key.upper()
        if value is None:
            flags[key] = {'value': 'N/A', 'status': 'unknown', 'label': lbl, 'unit': unit}
            return
        if low is not None and value < low:
            flags[key] = {'value': value, 'status': 'low', 'label': lbl,
                          'range': f'{low}–{high}', 'unit': unit}
            abnormal.append(lbl)
        elif high is not None and value > high:
            flags[key] = {'value': value, 'status': 'high', 'label': lbl,
                          'range': f'{low}–{high}', 'unit': unit}
            abnormal.append(lbl)
        else:
            flags[key] = {'value': value, 'status': 'normal', 'label': lbl,
                          'range': f'{low}–{high}', 'unit': unit}

    def flag_cat(key, value, normal_vals, label=None):
        lbl = label or key.upper()
        if value is None:
            flags[key] = {'value': 'N/A', 'status': 'unknown', 'label': lbl}
            return
        if value.lower() in normal_vals:
            flags[key] = {'value': value, 'status': 'normal', 'label': lbl}
        else:
            flags[key] = {'value': value, 'status': 'abnormal', 'label': lbl}
            abnormal.append(lbl)

    # Numeric checks (gender-aware)
    hb_range  = HEALTH_REF['hb_male']  if gender == 'Male' else HEALTH_REF['hb_female']
    rbc_range = HEALTH_REF['rbc_male'] if gender == 'Male' else HEALTH_REF['rbc_female']

    flag('hb',        data.get('hb'),        *hb_range,                 unit='g/dL',       label='Hemoglobin (Hb)')
    flag('rbc',       data.get('rbc'),        *rbc_range,                unit='Mill/cumm',  label='RBC Count')
    flag('platelets', data.get('platelets'),  *HEALTH_REF['platelets'],  unit='×10³/µL',   label='Platelet Count')

    # Categorical checks
    flag_cat('glucose',  data.get('glucose'),  HEALTH_CAT_NORMAL['glucose'],  label='Urine Glucose')
    flag_cat('bile',     data.get('bile'),     HEALTH_CAT_NORMAL['bile'],     label='Bile Pigment')
    flag_cat('pus',      data.get('pus'),      HEALTH_CAT_NORMAL['pus'],      label='Pus Cells')
    flag_cat('crystals', data.get('crystals'), HEALTH_CAT_NORMAL['crystals'], label='Crystals')

    n = len(abnormal)
    if n == 0:
        recommendation = 'All Clear'
        rec_level      = 'clear'
        rec_msg        = 'All parameters are within normal clinical range. No action required.'
    elif n <= 2:
        recommendation = 'Monitor / Retest'
        rec_level      = 'monitor'
        rec_msg        = f'{n} parameter(s) outside normal range. A follow-up test in 4–6 weeks is advisable.'
    else:
        recommendation = 'Consultation Recommended'
        rec_level      = 'consult'
        rec_msg        = f'{n} parameter(s) outside normal range. Please consult a physician for further evaluation.'

    return {
        'flags':          flags,
        'abnormal':       abnormal,
        'abnormal_count': n,
        'recommendation': recommendation,
        'rec_level':      rec_level,
        'rec_msg':        rec_msg,
    }


def append_health_record(data: dict, result: dict):
    """Save one row to student_health.csv."""
    import csv
    from datetime import datetime

    row = {
        'date':            datetime.now().strftime('%Y-%m-%d %H:%M'),
        'uhid':            data.get('uhid', ''),
        'gender':          data.get('gender', ''),
        'hb':              data.get('hb', ''),
        'rbc':             data.get('rbc', ''),
        'platelets':       data.get('platelets', ''),
        'bile':            data.get('bile', ''),
        'pus':             data.get('pus', ''),
        'crystals':        data.get('crystals', ''),
        'glucose':         data.get('glucose', ''),
        'abnormal_count':  result.get('abnormal_count', ''),
        'recommendation':  result.get('recommendation', ''),
    }

    fieldnames = list(row.keys())
    write_header = not os.path.exists(HEALTH_CSV)

    with open(HEALTH_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── Page route ────────────────────────────────

@app.route('/health-check')
def health_check():
    return render_template('health_check.html')


# ── API: parse PDF ────────────────────────────

@app.route('/api/parse-health-pdf', methods=['POST'])
def api_parse_health_pdf():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    f = request.files['file']
    if not f.filename.lower().endswith('.pdf'):
        return jsonify({'success': False, 'error': 'Only PDF files are supported'}), 400
    try:
        data = parse_suhrc_pdf(f.read())
        if data.get('error'):
            return jsonify({'success': False, 'error': data['error']}), 400
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        app.logger.error(f'Health PDF parse error: {e}')
        return jsonify({'success': False, 'error': str(e)}), 500


# ── API: run rule engine + save ───────────────

@app.route('/api/health-predict', methods=['POST'])
def api_health_predict():
    try:
        data   = request.get_json(force=True)
        result = run_health_rule_engine(data)
        append_health_record(data, result)
        return jsonify({'success': True, **result})
    except Exception as e:
        app.logger.error(f'Health predict error: {e}')
        return jsonify({'success': False, 'error': str(e)}), 500


# ── API: fetch saved dataset ──────────────────

@app.route('/api/health-records')
def api_health_records():
    import csv
    if not os.path.exists(HEALTH_CSV):
        return jsonify({'success': True, 'records': []})
    try:
        with open(HEALTH_CSV, newline='') as f:
            records = list(csv.DictReader(f))
        return jsonify({'success': True, 'records': records})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)