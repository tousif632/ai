import streamlit as st
import os, zipfile, tempfile, json, shutil
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model, save_model
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

os.makedirs("models", exist_ok=True)
st.set_page_config(page_title="AutoML Trainer", layout="centered")

if "page" not in st.session_state:
    st.session_state.page = "home"

# Navigation
if st.session_state.page == "home":
    st.title("🤖 AutoML Model Trainer")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📁 CSV-Based ML Model"):
            st.session_state.page = "csv"
            st.rerun()
    with col2:
        if st.button("🖼️ Image Classification Model"):
            st.session_state.page = "image"
            st.rerun()

# ---------------- CSV ML PAGE ------------------
elif st.session_state.page == "csv":
    st.title("📁 CSV-Based ML Model")
    if st.button("🔙 Back"):
        st.session_state.page = "home"
        st.rerun()

    uploaded_csv = st.file_uploader("Upload CSV Dataset", type="csv")
    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.write("Preview of Data:", df.head())

        target_col = st.selectbox("Select Target Column (Label)", df.columns)

        is_text_model = st.checkbox("🔤 This is a Text Classification Problem")
        text_col = st.selectbox("Select Text Column", [col for col in df.columns if col != target_col]) if is_text_model else None

        if st.button("Train CSV Model"):
            if is_text_model:
                X = df[text_col]
                y_enc = LabelEncoder()
                y = y_enc.fit_transform(df[target_col])
                model = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
                model.fit(X, y)
                acc = accuracy_score(y, model.predict(X))
                st.success(f"Text Classification Accuracy: {acc:.2%}")
                joblib.dump(model, "models/text_model.joblib")
                joblib.dump(y_enc, "models/text_encoder.joblib")

                with tempfile.TemporaryDirectory() as temp_dir:
                    shutil.copy("models/text_model.joblib", os.path.join(temp_dir, "text_model.joblib"))
                    shutil.copy("models/text_encoder.joblib", os.path.join(temp_dir, "text_encoder.joblib"))
                    with open(os.path.join(temp_dir, "app.py"), "w") as f:
                        f.write("""import streamlit as st\nimport joblib\nmodel = joblib.load('text_model.joblib')\nencoder = joblib.load('text_encoder.joblib')\nst.title('Text Classifier')\ntext = st.text_input('Enter text')\nif text:\n    pred = model.predict([text])[0]\n    label = encoder.inverse_transform([pred])[0]\n    st.success(f'Prediction: {label}')\n""")
                    with open(os.path.join(temp_dir, "requirements.txt"), "w") as f:
                        f.write("streamlit\nscikit-learn\njoblib")
                    shutil.make_archive("models/text_model_package", 'zip', temp_dir)
                with open("models/text_model_package.zip", "rb") as f:
                    st.download_button("Download Text Model ZIP", f.read(), file_name="text_model_package.zip", mime="application/zip")

            else:
                X = df.drop(columns=[target_col])
                y = df[target_col]

                for col in X.select_dtypes(include=['object']).columns:
                    X[col] = LabelEncoder().fit_transform(X[col])
                if y.dtype == 'object':
                    y = LabelEncoder().fit_transform(y)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier() if len(np.unique(y)) < 10 else RandomForestRegressor()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                if isinstance(model, RandomForestClassifier):
                    acc = accuracy_score(y_test, preds)
                    st.success(f"Classification Accuracy: {acc:.2%}")
                else:
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    st.success(f"Regression RMSE: {rmse:.2f}")
                joblib.dump(model, "models/csv_model.joblib")
                with open("models/csv_features.json", "w") as f:
                    json.dump({"features": list(X.columns)}, f)

                with tempfile.TemporaryDirectory() as temp_dir:
                    shutil.copy("models/csv_model.joblib", os.path.join(temp_dir, "csv_model.joblib"))
                    shutil.copy("models/csv_features.json", os.path.join(temp_dir, "csv_features.json"))
                    with open(os.path.join(temp_dir, "app.py"), "w") as f:
                        f.write("""import streamlit as st\nimport pandas as pd\nimport joblib\nimport json\nmodel = joblib.load('csv_model.joblib')\nwith open('csv_features.json') as f:\n schema = json.load(f)\nfile = st.file_uploader('Upload CSV', type='csv')\nif file:\n df = pd.read_csv(file)\n for col in df.select_dtypes(include=['object']).columns:\n  df[col] = df[col].astype('category').cat.codes\n df = df[schema['features']]\n preds = model.predict(df)\n st.write('Predictions:', preds)\n""")
                    with open(os.path.join(temp_dir, "requirements.txt"), "w") as f:
                        f.write("streamlit\npandas\nscikit-learn\njoblib")
                    shutil.make_archive("models/csv_model_package", 'zip', temp_dir)
                with open("models/csv_model_package.zip", "rb") as f:
                    st.download_button("Download CSV Model ZIP", f.read(), file_name="csv_model_package.zip", mime="application/zip")

# ---------------- IMAGE CLASSIFICATION PAGE ------------------
elif st.session_state.page == "image":
    st.title("🖼️ Image Classification Model")
    if st.button("🔙 Back"):
        st.session_state.page = "home"
        st.rerun()

    uploaded_zip = st.file_uploader("Upload ZIP of images", type="zip")
    if uploaded_zip:
        dataset_key = os.path.splitext(uploaded_zip.name)[0].replace(" ", "_")
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "dataset.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.read())
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)

            img_size = (150, 150)
            batch_size = 32
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
            train_gen = datagen.flow_from_directory(tmpdir, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='training')
            val_gen = datagen.flow_from_directory(tmpdir, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='validation')

            if train_gen.samples == 0:
                st.error("No images found. Ensure ZIP contains folders per class.")
            else:
                class_names = list(train_gen.class_indices.keys())
                num_classes = len(class_names)

                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(*img_size, 3)),
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(num_classes, activation='softmax')
                ])
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                history = model.fit(train_gen, validation_data=val_gen, epochs=5)
                val_acc = history.history['val_accuracy'][-1]
                st.success(f"Trained with Validation Accuracy: {val_acc:.2%}")

                model_path = f"models/{dataset_key}_model.h5"
                meta_path = f"models/{dataset_key}_meta.json"
                save_model(model, model_path)
                with open(meta_path, "w") as f:
                    json.dump({"class_names": class_names, "img_size": img_size}, f)

                zip_output_path = f"models/{dataset_key}_app.zip"
                with tempfile.TemporaryDirectory() as temp_dir:
                    shutil.copy(model_path, os.path.join(temp_dir, f"{dataset_key}_model.h5"))
                    shutil.copy(meta_path, os.path.join(temp_dir, f"{dataset_key}_meta.json"))
                    with open(os.path.join(temp_dir, "app.py"), "w") as f:
                        f.write(f"""import streamlit as st\nimport tensorflow as tf\nimport numpy as np\nfrom PIL import Image\nimport json\n\nmodel = tf.keras.models.load_model('{dataset_key}_model.h5')\nwith open('{dataset_key}_meta.json') as f:\n    meta = json.load(f)\nclass_names = meta['class_names']\nimg_size = tuple(meta['img_size'])\n\nst.title('Image Classifier')\nimg_file = st.file_uploader('Upload image', type=['jpg', 'jpeg', 'png'])\nif img_file:\n    img = Image.open(img_file).convert('RGB').resize(img_size)\n    arr = np.expand_dims(np.array(img) / 255.0, axis=0)\n    pred = model.predict(arr)\n    label = class_names[np.argmax(pred)]\n    st.image(img, caption=f'Prediction: {label}', use_column_width=True)\n""")
                    with open(os.path.join(temp_dir, "requirements.txt"), "w") as f:
                        f.write("streamlit\ntensorflow\npillow\nnumpy")
                    shutil.make_archive(zip_output_path.replace(".zip", ""), 'zip', temp_dir)
                with open(zip_output_path, "rb") as f:
                    st.download_button("Download Image Model ZIP", f.read(), file_name=os.path.basename(zip_output_path), mime="application/zip")

    st.markdown("### Try Live Prediction")
    test_img = st.file_uploader("Upload image to classify", type=["jpg", "jpeg", "png"], key="test")
    model_files = [f for f in os.listdir("models") if f.endswith("_model.h5")]
    if model_files:
        selected_model = st.selectbox("Choose trained model", model_files)
        if selected_model:
            base = selected_model.replace("_model.h5", "")
            model = load_model(f"models/{selected_model}")
            with open(f"models/{base}_meta.json") as f:
                meta = json.load(f)
            class_names = meta["class_names"]
            img_size = tuple(meta["img_size"])
            if test_img:
                img = Image.open(test_img).convert("RGB").resize(img_size)
                arr = np.expand_dims(np.array(img) / 255.0, axis=0)
                pred = model.predict(arr)
                st.image(img, caption=f"Prediction: {class_names[np.argmax(pred)]}", use_column_width=True)
