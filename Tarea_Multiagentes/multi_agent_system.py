import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

warnings.filterwarnings("ignore")

class Agent1Normalizer:
    """Agent 1: Normalizador. Cleans, imputes, scales, and encodes the dataset."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def process(self):
        print("--- [AGENTE 1] Normalizador ---")
        print(f"Cargando dataset: {self.filepath}")
        self.df = pd.read_csv(self.filepath)

        # 1. Limpieza
        self.df.drop_duplicates(inplace=True)
        print(f"Dimensiones después de eliminar duplicados: {self.df.shape}")

        # 2. Imputación
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].median())
            else:
                mode_val = self.df[col].mode()
                self.df[col] = self.df[col].fillna(mode_val[0] if not mode_val.empty else "Unknown")

        print("Imputación completada.")

        # Separar variables antes de escalar y codificar para mantener referencias si es necesario
        df_processed = self.df.copy()

        # 3. Codificación
        for col in df_processed.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            self.label_encoders[col] = le

        # 4. Escalar (escalaremos solo características numéricas seleccionadas luego si es necesario)
        # Para evitar problemas con la variable objetivo, dejaremos el escalado al pipeline del modelo o explícito
        num_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        # Asumiremos 'price_usd' como target si estamos intentando predecir, de lo contrario usaremos customer_rating
        # En este dataset ev_market_2026.csv, 'price_usd' es un buen target.
        if 'price_usd' in num_cols:
            features = [c for c in num_cols if c != 'price_usd']
            df_processed[features] = self.scaler.fit_transform(df_processed[features])
        
        print("Normalización, escalado y codificación terminada.")
        return df_processed, self.df # Return processed for Agente 2, raw for Agente 3


class Agent2Trainer:
    """Agent 2: Entrenador. Trains multiple models, applies validation, and selects the best."""
    def __init__(self, df, target_col='price_usd'):
        self.df = df
        self.target_col = target_col
        self.best_model = None
        self.best_name = ""
        self.best_r2 = -float('inf')

    def process(self):
        print("\n--- [AGENTE 2] Entrenador ---")
        if self.target_col not in self.df.columns:
            print(f"Columna objetivo '{self.target_col}' no encontrada. Abortando entrenamiento.")
            return None

        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        # Validación
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Datos divididos: Entrenamiento={X_train.shape[0]}, Prueba={X_test.shape[0]}")

        models = {
            "Regresión Lineal": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42, n_estimators=50),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=50)
        }

        print("Entrenando modelos y evaluando...")
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f" > {name}: R2 = {r2:.4f}, RMSE = {rmse:.2f}")

            if r2 > self.best_r2:
                self.best_r2 = r2
                self.best_model = model
                self.best_name = name

        print(f"*** El mejor modelo es {self.best_name} con un R2 de {self.best_r2:.4f} ***")
        return {"best_model": self.best_model, "best_name": self.best_name, "best_r2": self.best_r2}


class Agent3Communicator:
    """Agent 3: Comunicador. Uses Embeddings, Vector DB (FAISS) and Transformers to answer questions about the dataset."""
    def __init__(self, raw_df):
        self.df = raw_df
        self.embedding_model = None
        self.faiss_index = None
        self.corpus = []
        self.qa_pipeline = None

    def initialize_qa_system(self):
        print("\n--- [AGENTE 3] Comunicador ---")
        print("Inicializando modelo de embeddings (all-MiniLM-L6-v2)...")
        # Usamos un modelo ligero para los embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Generando corpus de documentos desde el dataset...")
        # Convertimos filas del dataset en texto natural para la base vectorial
        # Tomamos una muestra si el dataset es muy grande, o todo si es pequeño
        sample_df = self.df.sample(min(len(self.df), 1000)) 
        
        for index, row in sample_df.iterrows():
            # Crear un pequeño párrafo por cada fila
            text = (f"El auto {row.get('brand', '')} {row.get('model', '')} "
                    f"del año {row.get('year', '')} es un {row.get('body_type', '')} "
                    f"con una batería de {row.get('battery_capacity_kwh', '')} kWh, "
                    f"autonomía de {row.get('range_miles', '')} millas, "
                    f"y un precio de ${row.get('price_usd', '')}. "
                    f"Tiene una aceleración de 0 a 60 mph en {row.get('acceleration_0_60_mph', '')} segundos "
                    f"y es del país {row.get('country_of_origin', '')}.")
            self.corpus.append(text)

        print("Generando embeddings y creando Base Vectorial (FAISS)...")
        embeddings = self.embedding_model.encode(self.corpus, show_progress_bar=True, normalize_embeddings=True)
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension) # Usar similitud coseno
        self.faiss_index.add(np.array(embeddings, dtype='float32'))
        
        print("Inicializando modelo QA Transformer (google/flan-t5-large)...")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        self.qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

    def answer_question(self, question, top_k=3):
        if not self.faiss_index:
            print("El sistema QA no ha sido inicializado.")
            return

        print(f"\n[Usuario]: {question}")
        # Buscar contexto en FAISS
        question_emb = self.embedding_model.encode([question], normalize_embeddings=True)
        distances, indices = self.faiss_index.search(np.array(question_emb, dtype='float32'), top_k)
        
        # Recuperar documentos relevantes
        context = " ".join([self.corpus[idx] for idx in indices[0]])
        
        # Prompt para T5
        prompt = f"Basado en esta información: {context}\nResponde a la pregunta: {question}"
        
        # Generar respuesta
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.qa_model.generate(**inputs, max_length=150)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"[Agente 3]: {answer}")
        return answer


def run_multi_agent_system():
    dataset_path = r"c:\Users\kevin\OneDrive\Desktop\ia\Clase 8.6\ev_market_2026.csv"
    
    # === AGENTE 1 ===
    agent1 = Agent1Normalizer(dataset_path)
    processed_df, raw_df = agent1.process()
    
    # === AGENTE 2 ===
    agent2 = Agent2Trainer(processed_df, target_col='price_usd')
    agent2.process()
    
    # === AGENTE 3 ===
    agent3 = Agent3Communicator(raw_df)
    agent3.initialize_qa_system()
    
    print("\n¡Sistema listo! Puedes hacer preguntas sobre el dataset de autos eléctricos.")
    # Prueba de pregunta
    agent3.answer_question("¿Cuál es la batería del Volkswagen ID. Buzz?")
    agent3.answer_question("¿De qué país es el Toyota bZ Compact SUV?")

if __name__ == "__main__":
    run_multi_agent_system()
