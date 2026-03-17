# pip install streamlit pandas numpy scikit-learn plotly reportlab

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER
import datetime

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

[data-testid="collapsedControl"], section[data-testid="stSidebar"] { display:none; }
html, body, [class*="css"] { font-family:'Space Grotesk',sans-serif; }

.main-header {
    background: linear-gradient(135deg,#6C63FF 0%,#FF6584 50%,#43E97B 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    font-size:2.8rem; font-weight:700; text-align:center; letter-spacing:-1px; margin-bottom:0.2rem;
}
.sub-header {
    text-align:center; color:#888; font-size:0.9rem;
    margin-bottom:2rem; letter-spacing:2px; text-transform:uppercase;
}

/* ── Nav card buttons ── */
.nav-card-btn > button {
    background: linear-gradient(135deg,rgba(108,99,255,0.13),rgba(255,101,132,0.06)) !important;
    border: 1px solid rgba(108,99,255,0.3) !important;
    border-radius: 20px !important;
    padding: 2rem 1.5rem !important;
    width: 100% !important;
    min-height: 185px !important;
    color: inherit !important;
    font-weight: normal !important;
    letter-spacing: normal !important;
    white-space: pre-wrap !important;
    line-height: 1.6 !important;
    transition: box-shadow 0.22s, transform 0.22s, border-color 0.22s !important;
}
.nav-card-btn > button:hover {
    border-color: #6C63FF !important;
    box-shadow: 0 10px 36px rgba(108,99,255,0.35) !important;
    transform: translateY(-4px) !important;
    background: linear-gradient(135deg,rgba(108,99,255,0.2),rgba(255,101,132,0.1)) !important;
}
.nav-card-btn > button:active { transform: translateY(-1px) !important; }

.section-title {
    font-size:1.3rem; font-weight:600; color:#E0E0FF;
    border-left:4px solid #6C63FF; padding-left:1rem; margin:1.5rem 0 1rem;
}
.metric-card {
    background:linear-gradient(135deg,rgba(108,99,255,0.15),rgba(255,101,132,0.08));
    border:1px solid rgba(108,99,255,0.25); border-radius:16px;
    padding:1.2rem; text-align:center;
}
.metric-value { font-size:2.2rem; font-weight:700; color:#6C63FF; font-family:'JetBrains Mono',monospace; }
.metric-label { color:#aaa; font-size:0.78rem; text-transform:uppercase; letter-spacing:1.5px; margin-top:0.3rem; }

.success-box { background:rgba(67,233,123,0.1);  border:1px solid rgba(67,233,123,0.4);  border-radius:12px; padding:1rem 1.5rem; color:#43E97B; }
.warning-box { background:rgba(255,193,7,0.1);   border:1px solid rgba(255,193,7,0.35);  border-radius:12px; padding:1rem 1.5rem; color:#FFC107; }
.error-box   { background:rgba(255,101,132,0.1); border:1px solid rgba(255,101,132,0.4); border-radius:12px; padding:1rem 1.5rem; color:#FF6584; }
.info-box    { background:rgba(108,99,255,0.08); border:1px solid rgba(108,99,255,0.25); border-radius:12px; padding:1rem 1.5rem; color:#bbb; }

.badge-excellent { background:#43E97B22;color:#43E97B;border:1px solid #43E97B44;border-radius:20px;padding:4px 14px;font-size:0.85rem;font-weight:600; }
.badge-good      { background:#6C63FF22;color:#6C63FF;border:1px solid #6C63FF44;border-radius:20px;padding:4px 14px;font-size:0.85rem;font-weight:600; }
.badge-average   { background:#FFC10722;color:#FFC107;border:1px solid #FFC10744;border-radius:20px;padding:4px 14px;font-size:0.85rem;font-weight:600; }
.badge-needs     { background:#FF658422;color:#FF6584;border:1px solid #FF658444;border-radius:20px;padding:4px 14px;font-size:0.85rem;font-weight:600; }

.rec-item       { background:rgba(108,99,255,0.06);border-left:3px solid #6C63FF;padding:0.65rem 1rem;margin:0.4rem 0;border-radius:0 8px 8px 0;font-size:0.91rem; }
.rec-item-green { background:rgba(67,233,123,0.05);border-left:3px solid #43E97B;padding:0.65rem 1rem;margin:0.4rem 0;border-radius:0 8px 8px 0;font-size:0.91rem; }

.mapped-col  { background:rgba(108,99,255,0.08);border:1px solid rgba(108,99,255,0.3);border-radius:8px;padding:0.35rem 0.75rem;color:#bbb;font-family:'JetBrains Mono',monospace;font-size:0.8rem;display:inline-block;margin:0.2rem; }
.present-col { background:rgba(67,233,123,0.08);border:1px solid rgba(67,233,123,0.3); border-radius:8px;padding:0.4rem 0.8rem;color:#43E97B;font-family:'JetBrains Mono',monospace;font-size:0.82rem;display:inline-block;margin:0.2rem; }
.missing-col { background:rgba(255,193,7,0.08); border:1px dashed rgba(255,193,7,0.4);  border-radius:8px;padding:0.4rem 0.8rem;color:#FFC107;font-family:'JetBrains Mono',monospace;font-size:0.82rem;display:inline-block;margin:0.2rem; }

/* regular buttons */
.stButton > button {
    background:linear-gradient(135deg,#6C63FF,#FF6584) !important;
    color:white !important; border:none !important; border-radius:10px !important;
    font-weight:600 !important; font-family:'Space Grotesk',sans-serif !important;
    letter-spacing:0.5px !important; transition:all 0.3s ease !important; padding:0.5rem 1.5rem !important;
}
.stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 8px 25px rgba(108,99,255,0.4) !important; }
div[data-testid="metric-container"] { background:rgba(108,99,255,0.08);border:1px solid rgba(108,99,255,0.2);border-radius:12px;padding:1rem; }
</style>
""", unsafe_allow_html=True)

# ── Column alias map  ─────────────────────────────────────────
# Each entry: canonical_name -> list of accepted variants (lower-stripped)
COL_ALIASES = {
    'Student_ID':                 ['student_id','student id','id','studentid','roll_no','roll no','rollno',
                                   'student_name','studentname','student name','name','sname','s_name',
                                   'pupil','pupil_name','pupil name'],
    'Previous_Marks':             ['previous_marks','previous marks','prev_marks','prev marks',
                                   'last_marks','last marks','prior_marks','prior marks',
                                   'marks_before','old_marks','past_marks'],
    'Study_Hours_Per_Day':        ['study_hours_per_day','study hours per day','study_hours',
                                   'study hours','studyhours','daily_study','daily study',
                                   'hrs_study','hours_study','study_time'],
    'Work_Hours_Per_Week':        ['work_hours_per_week','work hours per week','work_hours',
                                   'work hours','workhours','weekly_work','weekly work',
                                   'parttime_hours','part_time_hours','job_hours'],
    'Sports_Hours_Per_Week':      ['sports_hours_per_week','sports hours per week','sports_hours',
                                   'sports hours','sportshours','physical_activity',
                                   'exercise_hours','activity_hours','fitness_hours'],
    'Mobile_Usage_Hours_Per_Day': ['mobile_usage_hours_per_day','mobile usage hours per day',
                                   'mobile_hours','mobile hours','mobilehours','phone_hours',
                                   'screen_time','screentime','phone_usage','mobile_usage',
                                   'device_hours'],
    'Attendance_Percentage':      ['attendance_percentage','attendance percentage','attendance',
                                   'attendance%','attend_pct','attend%','attendance_pct',
                                   'present_percentage','presence'],
    'Sleep_Hours_Per_Day':        ['sleep_hours_per_day','sleep hours per day','sleep_hours',
                                   'sleep hours','sleephours','daily_sleep','sleep_time',
                                   'hrs_sleep','hours_sleep'],
    'Family_Support':             ['family_support','family support','familysupport',
                                   'home_support','home support','parental_support',
                                   'parent_support','family_help'],
    'Internet_Access':            ['internet_access','internet access','internetaccess',
                                   'internet','net_access','wifi_access','has_internet',
                                   'online_access'],
}

FEATURE_COLS    = ['Previous_Marks','Study_Hours_Per_Day','Work_Hours_Per_Week',
                   'Sports_Hours_Per_Week','Mobile_Usage_Hours_Per_Day','Attendance_Percentage',
                   'Sleep_Hours_Per_Day','Family_Support','Internet_Access']
CATEGORY_MAP     = {0:'Needs Improvement',1:'Average',2:'Good',3:'Excellent'}
CATEGORY_MAP_INV = {v:k for k,v in CATEGORY_MAP.items()}
CATEGORY_COLORS  = {'Excellent':'#43E97B','Good':'#6C63FF','Average':'#FFC107','Needs Improvement':'#FF6584'}

DEFAULTS = {
    'Student_ID':'STU_AUTO','Previous_Marks':60,'Study_Hours_Per_Day':4.0,
    'Work_Hours_Per_Week':10,'Sports_Hours_Per_Week':3,'Mobile_Usage_Hours_Per_Day':3.0,
    'Attendance_Percentage':80,'Sleep_Hours_Per_Day':7.0,'Family_Support':'Medium','Internet_Access':'Yes',
}

# ── Session state ─────────────────────────────────────────────
for k, v in [('page','home'),('model_trained',False),('regression_model',None),
              ('classification_model',None),('scaler',None),
              ('batch_results',None),('selected_student',None)]:
    if k not in st.session_state:
        st.session_state[k] = v

def go_to(pg): st.session_state.page = pg

# ── Column detection ──────────────────────────────────────────
def detect_columns(df):
    """
    Returns (rename_map, missing_cols, mapping_notes).
    rename_map  : {original_col -> canonical_col}  for renaming
    missing_cols: canonical cols that could not be found at all
    mapping_notes: list of human-readable "found X as Y" strings
    """
    df_cols_lower = {c.lower().strip(): c for c in df.columns}
    rename_map    = {}
    missing_cols  = []
    notes         = []

    for canonical, aliases in COL_ALIASES.items():
        # 1. exact canonical match
        if canonical in df.columns:
            notes.append((canonical, canonical, 'exact'))
            continue
        # 2. alias match
        matched_orig = None
        matched_alias = None
        for alias in aliases:
            if alias in df_cols_lower:
                matched_orig  = df_cols_lower[alias]
                matched_alias = alias
                break
        if matched_orig:
            rename_map[matched_orig] = canonical
            notes.append((canonical, matched_orig, 'mapped'))
        else:
            missing_cols.append(canonical)
            notes.append((canonical, None, 'missing'))

    return rename_map, missing_cols, notes


def normalize_df(df):
    """Rename columns to canonical names and fill any missing with defaults."""
    rename_map, missing_cols, notes = detect_columns(df)
    df2 = df.rename(columns=rename_map)
    for mc in missing_cols:
        if mc == 'Student_ID':
            df2['Student_ID'] = [f'STU{str(i).zfill(4)}' for i in range(1, len(df2)+1)]
        else:
            df2[mc] = DEFAULTS.get(mc, 0)
    return df2, rename_map, missing_cols, notes

# ── Data / model helpers ──────────────────────────────────────
def generate_sample_data(n=500):
    np.random.seed(42)
    names = [f'STU{str(i).zfill(4)}' for i in range(1, n+1)]
    d = {
        'Student_ID': names,
        'Previous_Marks': np.random.randint(40,95,n),
        'Study_Hours_Per_Day': np.random.uniform(1,10,n).round(1),
        'Work_Hours_Per_Week': np.random.randint(0,30,n),
        'Sports_Hours_Per_Week': np.random.randint(0,15,n),
        'Mobile_Usage_Hours_Per_Day': np.random.uniform(1,8,n).round(1),
        'Attendance_Percentage': np.random.randint(60,100,n),
        'Sleep_Hours_Per_Day': np.random.uniform(4,9,n).round(1),
        'Family_Support': np.random.choice(['Low','Medium','High'],n),
        'Internet_Access': np.random.choice(['Yes','No'],n,p=[0.8,0.2]),
    }
    df = pd.DataFrame(d)
    df['Current_Marks'] = (
        df['Previous_Marks']*0.4 + df['Study_Hours_Per_Day']*3 +
        df['Attendance_Percentage']*0.2 - df['Mobile_Usage_Hours_Per_Day']*1.5 -
        df['Work_Hours_Per_Week']*0.3 + df['Sports_Hours_Per_Week']*0.5 +
        df['Sleep_Hours_Per_Day']*1.5 + np.random.normal(0,5,n)
    ).clip(0,100).round(1)
    df['Performance_Category'] = pd.cut(
        df['Current_Marks'], bins=[0,50,70,85,100],
        labels=['Needs Improvement','Average','Good','Excellent']
    )
    return df


def preprocess_data(df):
    d = df.copy()
    if d['Family_Support'].dtype == object:
        d['Family_Support'] = d['Family_Support'].map({'Low':0,'Medium':1,'High':2}).fillna(1)
    if d['Internet_Access'].dtype == object:
        d['Internet_Access'] = d['Internet_Access'].map({'Yes':1,'No':0}).fillna(1)
    if 'Performance_Category' in d.columns:
        d['Performance_Category_Encoded'] = d['Performance_Category'].map(CATEGORY_MAP_INV)
    return d


def _do_train(n=500):
    df = generate_sample_data(n)
    dp = preprocess_data(df)
    X = dp[FEATURE_COLS]; yr = dp['Current_Marks']; yc = dp['Performance_Category_Encoded']
    Xtr,Xte,ytr,yte,yctr,ycte = train_test_split(X,yr,yc,test_size=0.2,random_state=42)
    sc = StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
    rm = RandomForestRegressor(n_estimators=100,random_state=42,max_depth=10); rm.fit(Xtr_s,ytr)
    cm_ = GradientBoostingClassifier(n_estimators=100,random_state=42,max_depth=5); cm_.fit(Xtr_s,yctr)
    st.session_state.regression_model=rm; st.session_state.classification_model=cm_
    st.session_state.scaler=sc; st.session_state.model_trained=True
    return rm, cm_, sc, {
        'rmse':np.sqrt(mean_squared_error(yte,rm.predict(Xte_s))),
        'r2':r2_score(yte,rm.predict(Xte_s))
    }, {'accuracy':accuracy_score(ycte,cm_.predict(Xte_s))}


def ensure_model():
    """Silently auto-train if model not ready."""
    if not st.session_state.model_trained:
        _do_train(500)


def predict_single_raw(feat_dict):
    ensure_model()
    X = pd.DataFrame([feat_dict])[FEATURE_COLS]
    Xs = st.session_state.scaler.transform(X)
    pm = st.session_state.regression_model.predict(Xs)[0]
    pc = CATEGORY_MAP[st.session_state.classification_model.predict(Xs)[0]]
    return round(float(pm),1), pc


def predict_batch(df):
    ensure_model()
    dp = preprocess_data(df)
    X = dp[FEATURE_COLS]
    Xs = st.session_state.scaler.transform(X)
    pm = st.session_state.regression_model.predict(Xs).round(1)
    pc = [CATEGORY_MAP[c] for c in st.session_state.classification_model.predict(Xs)]
    return pm, pc

# ── Recommendations ───────────────────────────────────────────
GOOD_TIPS = [
    ("📖 Reading Habit",   "Add 20–30 min of daily reading to strengthen comprehension."),
    ("🗓️ Study Schedule",  "Build a weekly timetable and follow it consistently."),
    ("🤝 Group Study",     "Join a study group to tackle difficult topics together."),
    ("🎯 Goal Setting",    "Set specific weekly targets to stay motivated."),
    ("🧘 Stress Mgmt",    "Practise 10 min of mindfulness or deep breathing daily."),
    ("📝 Practice Tests",  "Regular mock tests boost long-term retention significantly."),
]

def get_recommendations(row, always_min=3):
    CHECKS = [
        ('Study_Hours_Per_Day',        lambda v:v<4,
         "📚 Study Hours",        lambda v:f"Increase to ≥4 hrs/day (currently {v} hrs). Study time is the biggest performance driver."),
        ('Mobile_Usage_Hours_Per_Day', lambda v:v>3,
         "📱 Mobile Usage",       lambda v:f"Reduce to ≤3 hrs/day (currently {v} hrs). Excess screen time hurts focus and retention."),
        ('Sleep_Hours_Per_Day',        lambda v:v<6.5,
         "😴 Sleep",              lambda v:f"Aim for 7–8 hrs/night (currently {v} hrs). Sleep is critical for memory consolidation."),
        ('Attendance_Percentage',      lambda v:v<80,
         "✅ Attendance",         lambda v:f"Bring above 80% (currently {v}%). Missing class directly hurts marks."),
        ('Sports_Hours_Per_Week',      lambda v:v<3,
         "⚽ Physical Activity",  lambda v:f"Increase to ≥3 hrs/week (currently {v} hrs). Exercise improves concentration."),
        ('Work_Hours_Per_Week',        lambda v:v>20,
         "💼 Work Hours",         lambda v:f"Try to keep below 20 hrs/week (currently {v} hrs) to protect study time."),
    ]
    failing = [(lbl, fn(float(row.get(col,0))), False)
               for col, chk, lbl, fn in CHECKS if chk(float(row.get(col,0)))]
    extra = max(0, always_min - len(failing))
    return failing + [(l,m,True) for l,m in GOOD_TIPS[:extra]]


def get_bulk_recs(df):
    recs = []
    def avg(c): return df[c].astype(float).mean()
    def pct(c,fn): return df[c].astype(float).apply(fn).mean()*100
    if avg('Mobile_Usage_Hours_Per_Day')>3:
        recs.append(f"📱 **Reduce Mobile Usage**: {pct('Mobile_Usage_Hours_Per_Day',lambda v:v>3):.0f}% exceed 3 hrs/day (avg {avg('Mobile_Usage_Hours_Per_Day'):.1f} hrs). Target ≤2 hrs/day.")
    if avg('Study_Hours_Per_Day')<4:
        recs.append(f"📚 **Increase Study Hours**: {pct('Study_Hours_Per_Day',lambda v:v<4):.0f}% study <4 hrs/day (avg {avg('Study_Hours_Per_Day'):.1f} hrs). Target 4–6 hrs/day.")
    if avg('Sports_Hours_Per_Week')<3:
        recs.append(f"⚽ **Increase Sports**: {pct('Sports_Hours_Per_Week',lambda v:v<3):.0f}% do <3 hrs/week (avg {avg('Sports_Hours_Per_Week'):.1f} hrs). Target 3–5 hrs/week.")
    if avg('Attendance_Percentage')<80:
        recs.append(f"✅ **Improve Attendance**: {pct('Attendance_Percentage',lambda v:v<80):.0f}% below 80% (avg {avg('Attendance_Percentage'):.1f}%). Target ≥85%.")
    if avg('Sleep_Hours_Per_Day')<6.5:
        recs.append(f"😴 **Better Sleep**: Average {avg('Sleep_Hours_Per_Day'):.1f} hrs/night. Students need 7–8 hrs.")
    return recs

# ── PDF ───────────────────────────────────────────────────────
def build_pdf_student(row, pm, pc, recs):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf,pagesize=A4,rightMargin=2*cm,leftMargin=2*cm,topMargin=2*cm,bottomMargin=2*cm)
    S = getSampleStyleSheet()
    T  = ParagraphStyle('T', fontSize=20,textColor=colors.HexColor('#6C63FF'),fontName='Helvetica-Bold',alignment=TA_CENTER,spaceAfter=4)
    Su = ParagraphStyle('Su',fontSize=9, textColor=colors.HexColor('#888888'),alignment=TA_CENTER,spaceAfter=16)
    Sh = ParagraphStyle('Sh',fontSize=13,textColor=colors.HexColor('#6C63FF'),fontName='Helvetica-Bold',spaceBefore=12,spaceAfter=7)
    Rp = ParagraphStyle('Rp',fontSize=10,leftIndent=10,spaceAfter=4,leading=15)
    Ft = ParagraphStyle('Ft',fontSize=8, textColor=colors.HexColor('#AAAAAA'),alignment=TA_CENTER)
    sid = str(row.get('Student_ID',''))
    story = [
        Paragraph("Student Performance Report",T),
        Paragraph(f"Generated {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}",Su),
        HRFlowable(width="100%",thickness=2,color=colors.HexColor('#6C63FF')),
        Spacer(1,10), Paragraph("Student Information",Sh),
    ]
    info = [
        ['Student',str(sid),'Predicted Marks',f"{pm:.1f}/100"],
        ['Category',str(pc),'Family Support',str(row.get('Family_Support',''))],
        ['Internet',str(row.get('Internet_Access','')),'',''],
    ]
    it=Table(info,colWidths=[4*cm,5*cm,5*cm,4*cm])
    it.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#6C63FF')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('BACKGROUND',(0,1),(-1,-1),colors.HexColor('#F8F8FF')),('GRID',(0,0),(-1,-1),.5,colors.HexColor('#CCCCCC')),
        ('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),10),
        ('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),('FONTNAME',(2,0),(2,-1),'Helvetica-Bold'),('PADDING',(0,0),(-1,-1),8),
    ]))
    story += [it, Spacer(1,10), Paragraph("Academic Habits",Sh)]
    def ok(v,cond): return 'OK' if cond else '!'
    habits=[
        ['Metric','Value','Benchmark','Status'],
        ['Previous Marks',str(row.get('Previous_Marks','')),'>=60',ok(row.get('Previous_Marks',0),float(row.get('Previous_Marks',0))>=60)],
        ['Study Hrs/Day',str(row.get('Study_Hours_Per_Day','')),'>=4 hrs',ok(0,float(row.get('Study_Hours_Per_Day',0))>=4)],
        ['Mobile Hrs/Day',str(row.get('Mobile_Usage_Hours_Per_Day','')),'<=3 hrs',ok(0,float(row.get('Mobile_Usage_Hours_Per_Day',0))<=3)],
        ['Sleep Hrs/Day',str(row.get('Sleep_Hours_Per_Day','')),'7-8 hrs',ok(0,6.5<=float(row.get('Sleep_Hours_Per_Day',0))<=9)],
        ['Sports Hrs/Wk',str(row.get('Sports_Hours_Per_Week','')),'>=3 hrs',ok(0,float(row.get('Sports_Hours_Per_Week',0))>=3)],
        ['Work Hrs/Wk',str(row.get('Work_Hours_Per_Week','')),'<=20 hrs',ok(0,float(row.get('Work_Hours_Per_Week',0))<=20)],
        ['Attendance',f"{row.get('Attendance_Percentage','')}%",'>=80%',ok(0,float(row.get('Attendance_Percentage',0))>=80)],
    ]
    ht=Table(habits,colWidths=[5.5*cm,3.5*cm,4*cm,3*cm])
    ht.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0D0D2B')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#FFFFFF'),colors.HexColor('#F5F5FF')]),
        ('GRID',(0,0),(-1,-1),.5,colors.HexColor('#DDDDDD')),('FONTSIZE',(0,0),(-1,-1),10),
        ('PADDING',(0,0),(-1,-1),8),('ALIGN',(1,0),(-1,-1),'CENTER'),
    ]))
    story+=[ht,Spacer(1,12),Paragraph("Recommendations",Sh)]
    for lbl,txt,_ in recs: story.append(Paragraph(f"<b>{lbl}:</b> {txt}",Rp))
    story+=[Spacer(1,14),HRFlowable(width="100%",thickness=1,color=colors.HexColor('#CCCCCC')),
            Spacer(1,6),Paragraph("Student Academic Performance Analytics | © 2026 Yogavannan",Ft)]
    doc.build(story); buf.seek(0); return buf.read()


def build_pdf_batch(df):
    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=A4,rightMargin=1.5*cm,leftMargin=1.5*cm,topMargin=2*cm,bottomMargin=2*cm)
    S=getSampleStyleSheet()
    T  = ParagraphStyle('T', fontSize=20,textColor=colors.HexColor('#6C63FF'),fontName='Helvetica-Bold',alignment=TA_CENTER,spaceAfter=4)
    Su = ParagraphStyle('Su',fontSize=9, textColor=colors.HexColor('#888888'),alignment=TA_CENTER,spaceAfter=14)
    Sh = ParagraphStyle('Sh',fontSize=13,textColor=colors.HexColor('#6C63FF'),fontName='Helvetica-Bold',spaceBefore=12,spaceAfter=7)
    Rp = ParagraphStyle('Rp',fontSize=10,leftIndent=10,spaceAfter=4,leading=15)
    Ft = ParagraphStyle('Ft',fontSize=8, textColor=colors.HexColor('#AAAAAA'),alignment=TA_CENTER)
    story=[
        Paragraph("Batch Student Performance Report",T),
        Paragraph(f"Total: {len(df)} students | {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}",Su),
        HRFlowable(width="100%",thickness=2,color=colors.HexColor('#6C63FF')),Spacer(1,10),
        Paragraph("Summary",Sh),
    ]
    cc=df['Predicted_Category'].value_counts()
    sumdata=[['Category','Count','%']]+[
        [cat,str(cc.get(cat,0)),f"{cc.get(cat,0)/len(df)*100:.1f}%"]
        for cat in ['Excellent','Good','Average','Needs Improvement']
    ]+[['Avg Predicted Marks',f"{df['Predicted_Marks'].mean():.1f}",'']]
    st2=Table(sumdata,colWidths=[8*cm,4*cm,4*cm])
    st2.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#6C63FF')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#FFFFFF'),colors.HexColor('#F0F0FF')]),
        ('GRID',(0,0),(-1,-1),.5,colors.HexColor('#CCCCCC')),('FONTSIZE',(0,0),(-1,-1),10),
        ('PADDING',(0,0),(-1,-1),8),('ALIGN',(1,0),(-1,-1),'CENTER'),
    ]))
    story+=[st2,Spacer(1,10),Paragraph("Class-Wide Recommendations",Sh)]
    for r in get_bulk_recs(df): story.append(Paragraph(f"• {r.replace('**','')}",Rp))
    story+=[Spacer(1,10),Paragraph("Individual Results",Sh)]
    hdr=['Student','Pred. Marks','Category','Study Hrs','Mobile Hrs','Sports Hrs','Attend%']
    tdata=[hdr]+[
        [str(r['Student_ID']),f"{r['Predicted_Marks']:.1f}",str(r['Predicted_Category']),
         str(r['Study_Hours_Per_Day']),str(r['Mobile_Usage_Hours_Per_Day']),
         str(r['Sports_Hours_Per_Week']),f"{r['Attendance_Percentage']}%"]
        for _,r in df.iterrows()
    ]
    cat_bg={'Excellent':'#E8FFF0','Good':'#EEEEFF','Average':'#FFFBEA','Needs Improvement':'#FFF0F3'}
    rstyles=[('BACKGROUND',(0,i),(-1,i),colors.HexColor(cat_bg.get(str(df.iloc[i-1]['Predicted_Category']),'#FFF')))
             for i in range(1,len(df)+1)]
    bt=Table(tdata,colWidths=[3*cm,2.5*cm,4*cm,2.5*cm,2.5*cm,2.5*cm,2.5*cm],repeatRows=1)
    bt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0D0D2B')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('GRID',(0,0),(-1,-1),.5,colors.HexColor('#CCCCCC')),
        ('FONTSIZE',(0,0),(-1,-1),8),('PADDING',(0,0),(-1,-1),5),('ALIGN',(1,0),(-1,-1),'CENTER'),*rstyles
    ]))
    story+=[bt,Spacer(1,14),HRFlowable(width="100%",thickness=1,color=colors.HexColor('#CCCCCC')),
            Spacer(1,6),Paragraph("Student Academic Performance Analytics | © 2026 Yogavannan",Ft)]
    doc.build(story); buf.seek(0); return buf.read()

# ═══════════════════════════════════════════════════════════════
#  PAGES
# ═══════════════════════════════════════════════════════════════
page = st.session_state.page

# ─────────────────────── HOME ───────────────────────
if page == 'home':
    if "nav" in st.query_params:
        st.query_params.clear()

    st.markdown('<h1 class="main-header">🎓 Student Performance Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning · Predictions · Insights · Recommendations</p>', unsafe_allow_html=True)

    status  = "✅ Model ready" if st.session_state.model_trained else "⚡ Model auto-trains on first use"
    box_cls = "success-box" if st.session_state.model_trained else "info-box"
    st.markdown(f'<div class="{box_cls}" style="text-align:center;">{status}</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Style each button by its data-testid key attribute Streamlit adds
    st.markdown("""
    <style>
    button[kind="secondary"][data-testid="baseButton-secondary"] { text-align:center; }
    div:has(> button[key="nav_train"]) > button,
    div:has(> button[key="nav_batch"]) > button,
    div:has(> button[key="nav_single"]) > button {
        background: linear-gradient(135deg,rgba(108,99,255,0.13),rgba(255,101,132,0.06)) !important;
        border: 1px solid rgba(108,99,255,0.3) !important;
        border-radius: 20px !important;
        min-height: 185px !important;
        color: inherit !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 400 !important;
        letter-spacing: normal !important;
        white-space: pre-wrap !important;
        line-height: 1.7 !important;
        transition: box-shadow 0.22s, transform 0.22s, border-color 0.22s !important;
    }
    div:has(> button[key="nav_train"]) > button:hover,
    div:has(> button[key="nav_batch"]) > button:hover,
    div:has(> button[key="nav_single"]) > button:hover {
        border-color: #6C63FF !important;
        box-shadow: 0 10px 36px rgba(108,99,255,0.35) !important;
        transform: translateY(-4px) !important;
        background: linear-gradient(135deg,rgba(108,99,255,0.2),rgba(255,101,132,0.1)) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("🔧\n\nTrain Model\n\nTrain ML models on sample data and see feature importance.",
                     key="nav_train", use_container_width=True):
            go_to('train'); st.rerun()

    with c2:
        if st.button("📁\n\nBatch Upload\n\nUpload any CSV — columns auto-detected, predictions instant.",
                     key="nav_batch", use_container_width=True):
            go_to('batch'); st.rerun()

    with c3:
        if st.button("✍️\n\nSingle Entry\n\nEnter one student's data — live predictions as you type.",
                     key="nav_single", use_container_width=True):
            go_to('single'); st.rerun()

    st.markdown("---")
    st.markdown('<div class="section-title">Accepted Column Names (any variant works)</div>', unsafe_allow_html=True)
    for canonical, aliases in COL_ALIASES.items():
        top3 = ', '.join(aliases[:4])
        st.markdown(f'<span class="present-col">{canonical}</span>'
                    f'<span style="color:#555;font-size:0.75rem;"> also accepts: {top3} …</span><br>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Sample CSV Template</div>', unsafe_allow_html=True)
    sample_df = generate_sample_data(5)
    st.dataframe(sample_df[['Student_ID']+FEATURE_COLS].head(), use_container_width=True)
    st.download_button("📥 Download Sample CSV",
                       data=sample_df[['Student_ID']+FEATURE_COLS].to_csv(index=False),
                       file_name="student_template.csv", mime="text/csv")

    st.markdown("<div style='text-align:center;color:#444;font-size:0.78rem;'>"
                "Student Academic Performance  ©  Yoga 2026</div>",
                unsafe_allow_html=True)


# ─────────────────────── TRAIN ───────────────────────
elif page == 'train':
    if st.button("← Back to Home"): go_to('home'); st.rerun()

    st.markdown('<h2 class="section-title">🔧 Train Machine Learning Models</h2>', unsafe_allow_html=True)
    st.write("Trains **Random Forest Regression** (marks) and **Gradient Boosting Classification** (category).")

    n_samples = st.slider("Training samples", 200, 2000, 500, 50)
    if st.button("🚀 Train Models Now"):
        with st.spinner("Training..."):
            rm, cm_, sc, reg_m, clf_m = _do_train(n_samples)
        st.markdown('<div class="success-box">✅ Models trained! Batch Upload and Single Entry are ready.</div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("R² Score", f"{reg_m['r2']:.4f}")
        c2.metric("RMSE", f"{reg_m['rmse']:.2f}")
        c3.metric("Classification Accuracy", f"{clf_m['accuracy']*100:.1f}%")
        fi_df = pd.DataFrame({'Feature':FEATURE_COLS,'Importance':rm.feature_importances_}).sort_values('Importance')
        fig = px.bar(fi_df,x='Importance',y='Feature',orientation='h',
                     color='Importance',color_continuous_scale='Purples',template='plotly_dark')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',showlegend=False,height=350)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────── BATCH ───────────────────────
elif page == 'batch':
    if st.button("← Back to Home"): go_to('home'); st.rerun()

    st.markdown('<h2 class="section-title">📁 Batch Upload & Predictions</h2>', unsafe_allow_html=True)

    # No "must train first" block — ensure_model() handles it silently inside predict_batch()

    uploaded = st.file_uploader("Upload CSV file (any column names — auto-detected)", type=['csv'])

    if uploaded:
        raw_df = pd.read_csv(uploaded)

        # ── Detect & normalise columns ──────────────────────
        df_norm, rename_map, missing_cols, notes = normalize_df(raw_df)

        st.markdown('<div class="section-title">📋 Column Detection</div>', unsafe_allow_html=True)
        exact_notes    = [(c,o,t) for c,o,t in notes if t=='exact']
        mapped_notes   = [(c,o,t) for c,o,t in notes if t=='mapped']
        missing_notes  = [(c,o,t) for c,o,t in notes if t=='missing']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**✅ Exact match:**")
            st.markdown("".join([f'<span class="present-col">{c}</span>' for c,_,_ in exact_notes]) or "<i style='color:#555'>none</i>",
                        unsafe_allow_html=True)
        with col2:
            st.markdown("**🔄 Auto-mapped:**")
            st.markdown("".join([f'<span class="mapped-col">{o} → {c}</span>' for c,o,_ in mapped_notes]) or "<i style='color:#555'>none</i>",
                        unsafe_allow_html=True)
        with col3:
            if missing_notes:
                st.markdown("**⚠️ Missing (default used):**")
                st.markdown("".join([f'<span class="missing-col">{c}</span>' for c,_,_ in missing_notes]),
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✅ All columns found!</div>', unsafe_allow_html=True)

        if missing_notes:
            st.markdown(
                f'<div class="warning-box">⚠️ {len(missing_notes)} column(s) not found in your CSV — '
                f'default values used so predictions can still run. Accuracy may be reduced for those fields.</div>',
                unsafe_allow_html=True)

        # ── Data preview ────────────────────────────────────
        st.markdown('<div class="section-title">📊 Uploaded Data</div>', unsafe_allow_html=True)
        show_cols = ['Student_ID'] + [c for c in FEATURE_COLS if c in df_norm.columns]
        st.dataframe(df_norm[show_cols], use_container_width=True)

        # ── Auto-predict immediately ─────────────────────────
        pm, pc = predict_batch(df_norm)
        df_res = df_norm.copy()
        df_res['Predicted_Marks']    = pm
        df_res['Predicted_Category'] = pc
        st.session_state.batch_results    = df_res
        st.session_state.selected_student = None

    if st.session_state.batch_results is not None:
        df_res = st.session_state.batch_results

        # Summary
        st.markdown('<div class="section-title">📈 Prediction Summary</div>', unsafe_allow_html=True)
        cc = df_res['Predicted_Category'].value_counts()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Avg Predicted Marks", f"{df_res['Predicted_Marks'].mean():.1f}")
        c2.metric("⭐ Excellent",         cc.get('Excellent',0))
        c3.metric("✅ Good",              cc.get('Good',0))
        c4.metric("⚠️ Needs Improvement", cc.get('Needs Improvement',0))

        cat_df = df_res['Predicted_Category'].value_counts().reset_index()
        cat_df.columns=['Category','Count']
        fig_pie = px.pie(cat_df,values='Count',names='Category',color='Category',
                         color_discrete_map=CATEGORY_COLORS,hole=0.55,template='plotly_dark')
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)',height=300,legend=dict(font=dict(color='#ccc')))
        st.plotly_chart(fig_pie, use_container_width=True)

        # Class recs
        st.markdown('<div class="section-title">💡 Class-Wide Recommendations</div>', unsafe_allow_html=True)
        for r in get_bulk_recs(df_res):
            st.markdown(f'<div class="rec-item">{r}</div>', unsafe_allow_html=True)

        # Results table
        st.markdown('<div class="section-title">📋 All Student Results</div>', unsafe_allow_html=True)
        disp_cols = ['Student_ID','Predicted_Marks','Predicted_Category',
                     'Study_Hours_Per_Day','Mobile_Usage_Hours_Per_Day',
                     'Sports_Hours_Per_Week','Attendance_Percentage']
        disp_cols = [c for c in disp_cols if c in df_res.columns]
        st.dataframe(df_res[disp_cols], use_container_width=True, height=300)

        # Student detail
        student_list = df_res['Student_ID'].astype(str).tolist()
        sid = st.selectbox("🔍 Select a student to view details:", ['-- Select --'] + student_list)
        if sid != '-- Select --':
            st.session_state.selected_student = sid

        if st.session_state.selected_student and st.session_state.selected_student != '-- Select --':
            srow = df_res[df_res['Student_ID'].astype(str) == str(st.session_state.selected_student)].iloc[0]
            spm  = srow['Predicted_Marks']
            spc  = srow['Predicted_Category']
            srec = get_recommendations(srow)
            bc   = {'Excellent':'badge-excellent','Good':'badge-good',
                    'Average':'badge-average','Needs Improvement':'badge-needs'}.get(spc,'badge-average')

            st.markdown(f'<div class="section-title">👤 {st.session_state.selected_student}</div>', unsafe_allow_html=True)
            col1, col2 = st.columns([1,2])
            with col1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{spm:.1f}</div>
                    <div class="metric-label">Predicted Marks</div>
                    <br><span class="{bc}">{spc}</span>
                </div>""", unsafe_allow_html=True)
            with col2:
                cats = ['Study','Sleep','Attendance','Sports']
                vals = [
                    min(float(srow.get('Study_Hours_Per_Day',4))/10*100,100),
                    min(float(srow.get('Sleep_Hours_Per_Day',7))/9*100,100),
                    float(srow.get('Attendance_Percentage',80)),
                    min(float(srow.get('Sports_Hours_Per_Week',3))/15*100,100),
                ]
                fig_r = go.Figure(go.Scatterpolar(r=vals+[vals[0]],theta=cats+[cats[0]],
                    fill='toself',line_color='#6C63FF',fillcolor='rgba(108,99,255,0.2)'))
                fig_r.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),
                    paper_bgcolor='rgba(0,0,0,0)',height=230,margin=dict(l=30,r=30,t=20,b=20),showlegend=False)
                st.plotly_chart(fig_r, use_container_width=True)

            st.markdown("**Personalized Recommendations:**")
            for lbl, txt, is_tip in srec:
                cls = "rec-item-green" if is_tip else "rec-item"
                st.markdown(f'<div class="{cls}"><b>{lbl}:</b> {txt}</div>', unsafe_allow_html=True)

            pdf_b = build_pdf_student(srow.to_dict(), spm, spc, srec)
            st.download_button(f"📄 Download {st.session_state.selected_student} PDF",
                               data=pdf_b, file_name=f"{st.session_state.selected_student}_report.pdf",
                               mime="application/pdf")

        # Downloads
        st.markdown("---")
        dl1, dl2 = st.columns(2)
        with dl1:
            if st.button("📄 Generate Full Batch PDF"):
                with st.spinner("Building PDF..."):
                    bpdf = build_pdf_batch(df_res)
                st.download_button("📥 Download Batch PDF", data=bpdf,
                                   file_name="batch_report.pdf", mime="application/pdf")
        with dl2:
            st.download_button("📥 Download Results CSV",
                               data=df_res.to_csv(index=False),
                               file_name="predictions.csv", mime="text/csv")


# ─────────────────────── SINGLE ENTRY ───────────────────────
elif page == 'single':
    if st.button("← Back to Home"): go_to('home'); st.rerun()

    st.markdown('<h2 class="section-title">✍️ Single Student Entry</h2>', unsafe_allow_html=True)
    st.write("Adjust any value — **prediction and recommendations update live**. No submit button needed.")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown('<div class="section-title">📝 Enter Student Data</div>', unsafe_allow_html=True)
        student_id  = st.text_input("Student Name / ID", "Student 001")
        prev_marks  = st.number_input("Previous Marks (0–100)",               min_value=0,   max_value=100,  value=75,  step=1)
        study_hours = st.number_input("Study Hours per Day (0–12)",            min_value=0.0, max_value=12.0, value=5.0, step=0.5)
        attendance  = st.number_input("Attendance Percentage (0–100)",         min_value=0,   max_value=100,  value=85,  step=1)
        work_hours  = st.number_input("Work Hours per Week (0–40)",            min_value=0,   max_value=40,   value=10,  step=1)
        sports_hrs  = st.number_input("Sports Hours per Week (0–20)",          min_value=0,   max_value=20,   value=5,   step=1)
        mobile_hrs  = st.number_input("Mobile Usage Hours per Day (0–12)",     min_value=0.0, max_value=12.0, value=3.0, step=0.5)
        sleep_hrs   = st.number_input("Sleep Hours per Day (0–12)",            min_value=0.0, max_value=12.0, value=7.0, step=0.5)
        fam_support = st.number_input("Family Support  (0=Low  1=Medium  2=High)", min_value=0, max_value=2, value=1, step=1)
        internet    = st.number_input("Internet Access (1=Yes  0=No)",         min_value=0,   max_value=1,    value=1,   step=1)

    feat = {
        'Previous_Marks':prev_marks,'Study_Hours_Per_Day':study_hours,
        'Work_Hours_Per_Week':work_hours,'Sports_Hours_Per_Week':sports_hrs,
        'Mobile_Usage_Hours_Per_Day':mobile_hrs,'Attendance_Percentage':attendance,
        'Sleep_Hours_Per_Day':sleep_hrs,'Family_Support':fam_support,'Internet_Access':internet,
    }
    ensure_model()
    pred_marks, pred_cat = predict_single_raw(feat)
    bc    = {'Excellent':'badge-excellent','Good':'badge-good',
             'Average':'badge-average','Needs Improvement':'badge-needs'}.get(pred_cat,'badge-average')
    grade = ('A+' if pred_marks>=90 else 'A' if pred_marks>=85 else
             'B'  if pred_marks>=70 else 'C' if pred_marks>=50 else 'D')

    with right:
        st.markdown('<div class="section-title">🎯 Live Prediction</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="metric-card" style="margin-bottom:1rem;">
            <div class="metric-value">{pred_marks:.1f}</div>
            <div class="metric-label">Predicted Marks</div>
            <br>
            <span class="{bc}" style="font-size:1rem;">{pred_cat}</span>
            &nbsp;&nbsp;
            <span style="color:#aaa;font-size:1.4rem;font-family:'JetBrains Mono',monospace;font-weight:700;">{grade}</span>
        </div>""", unsafe_allow_html=True)

        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=pred_marks,
            domain={'x':[0,1],'y':[0,1]},
            number={'font':{'color':'#6C63FF','size':28}},
            gauge={'axis':{'range':[0,100],'tickcolor':'#666'},'bar':{'color':'#6C63FF'},
                   'steps':[{'range':[0,50],'color':'rgba(255,101,132,0.25)'},
                             {'range':[50,70],'color':'rgba(255,193,7,0.25)'},
                             {'range':[70,85],'color':'rgba(108,99,255,0.25)'},
                             {'range':[85,100],'color':'rgba(67,233,123,0.25)'}]}
        ))
        fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)',font_color='#ccc',
                             height=200,margin=dict(l=15,r=15,t=15,b=10))
        st.plotly_chart(fig_g, use_container_width=True)

        row_dict = {'Study_Hours_Per_Day':study_hours,'Mobile_Usage_Hours_Per_Day':mobile_hrs,
                    'Sleep_Hours_Per_Day':sleep_hrs,'Work_Hours_Per_Week':work_hours,
                    'Attendance_Percentage':attendance,'Sports_Hours_Per_Week':sports_hrs}
        live_recs = get_recommendations(row_dict, always_min=3)

        st.markdown('<div class="section-title">💡 Personalized Recommendations</div>', unsafe_allow_html=True)
        st.caption(f"{len(live_recs)} recommendations  •  🟣 = improve  |  🟢 = good habit tip")
        for lbl, txt, is_tip in live_recs:
            cls = "rec-item-green" if is_tip else "rec-item"
            st.markdown(f'<div class="{cls}"><b>{lbl}:</b> {txt}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        row_pdf = {**feat,'Student_ID':student_id,
                   'Family_Support':{0:'Low',1:'Medium',2:'High'}.get(int(fam_support),'Medium'),
                   'Internet_Access':'Yes' if internet==1 else 'No'}
        pdf_b = build_pdf_student(row_pdf, pred_marks, pred_cat, live_recs)
        st.download_button("📄 Download Student Report (PDF)", data=pdf_b,
                           file_name=f"{student_id.replace(' ','_')}_report.pdf", mime="application/pdf")

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div style='text-align:center;color:#444;font-size:0.78rem;padding:0.5rem 0;'>"
            "Student Academic Performance Analytics  © 2026 "
            "</div>", unsafe_allow_html=True)