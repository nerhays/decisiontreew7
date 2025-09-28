# library
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import io

# pengaturan judul dan halaman
st.set_page_config(page_title="Decision Tree App", layout="wide")
st.title("Decision Tree — UI/UX (Streamlit)")
st.markdown("Ikuti langkah: upload dataset (atau gunakan default), pilih kolom target, latih model, dan lakukan prediksi.")

# Sidebar: upload atau input dataset
st.sidebar.header("Pengaturan Dataset")
uploaded_file = st.sidebar.file_uploader("Upload file Excel (.xlsx)", type=['xlsx'])

# Cek apakah dataset sudah ada di session_state
if 'df' not in st.session_state:
    if uploaded_file is not None:
        st.session_state['df'] = pd.read_excel(uploaded_file)
    else:
        if st.sidebar.button("Gunakan dataset default"):
            st.session_state['df'] = pd.read_excel('BlaBla.xlsx')
else:
    # Jika user upload lagi, update dataset
    if uploaded_file is not None:
        st.session_state['df'] = pd.read_excel(uploaded_file)

# Kalau dataset belum ada, stop
if 'df' not in st.session_state:
    st.info("Upload dataset atau tekan tombol di sidebar untuk pakai dataset default.")
    st.stop()

# Pakai dataset dari session_state
df = st.session_state['df']

# preview data set
st.subheader("Preview Data")
st.dataframe(df.head(100))

st.subheader("Info & Statistik")
buf = io.StringIO()
df.info(buf=buf)
info_str = buf.getvalue()
st.text(info_str)
st.write(df.describe(include='all').T)

# Pilih target
st.subheader("Pilih kolom Target / Label")
target_col = st.selectbox("Pilih kolom target", options=df.columns.tolist(), index=len(df.columns)-1)
st.write("Target terpilih:", target_col)

# Pilih fitur (dengan default semua kecuali target)
features = st.multiselect("Pilih fitur (kosong = semua kolom kecuali target)", options=[c for c in df.columns if c!=target_col])
if not features:
    X_df = df.drop(columns=[target_col])
else:
    X_df = df[features]
y_ser = df[target_col]

st.write("Fitur akan digunakan:", X_df.columns.tolist())

# Preprocessing sederhana
def preprocess_X(X):
    Xc = X.copy()
    # numeric fill
    num_cols = Xc.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        Xc[c] = Xc[c].fillna(Xc[c].median())
    # categorical fill
    cat_cols = Xc.select_dtypes(exclude=[np.number]).columns
    for c in cat_cols:
        Xc[c] = Xc[c].fillna('<<MISSING>>')
    # label encode categorical
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        Xc[c] = le.fit_transform(Xc[c].astype(str))
        encoders[c] = le
    return Xc, encoders

# Encode target if needed
def encode_target(y):
    if y.dtype=='O' or str(y.dtype).startswith('category'):
        le = LabelEncoder()
        y_enc = le.fit_transform(y.fillna('<<MISSING>>').astype(str))
        return y_enc, le
    else:
        return y.fillna(y.median()).astype(int), None

if st.button("Train Decision Tree"):
    with st.spinner("Melatih model..."):
        X_enc, encs = preprocess_X(X_df)
        y_enc, target_le = encode_target(y_ser)
        X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=0.2, random_state=42, stratify=y_enc if len(np.unique(y_enc))>1 else None)
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Training selesai. Akurasi pada test set: {acc:.4f}")
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)

        # Visualize tree
        st.subheader("Visualisasi Decision Tree")
        fig, ax = plt.subplots(figsize=(12,6))
        class_names = [str(c) for c in (target_le.classes_ if target_le is not None else np.unique(y_enc))]
        plot_tree(clf, feature_names=X_enc.columns, class_names=class_names, filled=True, rounded=True, fontsize=8, ax=ax)
        st.pyplot(fig)

        # Save model to session state for predict
        st.session_state['clf'] = clf
        st.session_state['encs'] = encs
        st.session_state['X_columns'] = list(X_enc.columns)
        st.session_state['target_le'] = target_le
        st.session_state['X_sample_dtypes'] = X_df.dtypes.to_dict()
        st.success("Model tersimpan untuk prediksi manual (di bawah).")

# Manual input & prediksi
st.subheader("Prediksi Manual (menggunakan model yang terakhir dilatih)")
if 'clf' in st.session_state:
    clf = st.session_state['clf']
    encs = st.session_state['encs']
    X_columns = st.session_state['X_columns']
    sample_dtypes = st.session_state['X_sample_dtypes']
    input_vals = {}
    st.write("Masukkan nilai untuk setiap fitur:")
    cols = st.columns(2)
    for i, col in enumerate(X_columns):
        dtype = sample_dtypes.get(col, 'object')
        with cols[i % 2]:
            if str(dtype).startswith('int') or str(dtype).startswith('float'):
                val = st.number_input(f"{col}", value=0.0)
            else:
                val = st.text_input(f"{col}", value="")
            input_vals[col] = val

    if st.button("Predict"):
        # build single-row df and encode
        row = pd.DataFrame([input_vals])
        # fill / convert types
        for c in row.columns:
            if c in encs:
                le = encs[c]
                v = str(row.at[0,c]) if row.at[0,c] != "" else "<<MISSING>>"
                # if value unseen for label encoder -> add handling: map to -1
                if v in le.classes_:
                    row[c] = le.transform([v])[0]
                else:
                    # unseen -> try to append or send fallback
                    row[c] = -1
            else:
                try:
                    row[c] = pd.to_numeric(row[c])
                except:
                    # fallback: try label encode on the fly
                    try:
                        row[c] = float(row[c])
                    except:
                        row[c] = 0
        pred = clf.predict(row[X_columns])
        if st.session_state['target_le'] is not None:
            pred_label = st.session_state['target_le'].inverse_transform(pred)[0]
        else:
            pred_label = str(pred[0])
        st.success(f"Prediksi: {pred_label}")

else:
    st.info("Latih model terlebih dahulu (klik tombol Train).")
