"""
Script para verificar que el entorno esté correctamente configurado
"""
import sys
from pathlib import Path

def check_python_version():
    """Verifica la versión de Python."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ requerido")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Verifica que las dependencias estén instaladas."""
    required = [
        'cv2', 'mediapipe', 'numpy', 'pandas', 'sklearn',
        'xgboost', 'joblib', 'matplotlib', 'seaborn'
    ]
    missing = []
    
    for dep in required:
        try:
            if dep == 'cv2':
                import cv2
            elif dep == 'sklearn':
                import sklearn
            else:
                __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} no instalado")
            missing.append(dep)
    
    return len(missing) == 0

def check_structure():
    """Verifica la estructura de directorios."""
    from src.config import PROJECT_ROOT, POSES_DIR, VIDEOS_DIR
    
    # Desde Entrega 3, las rutas son relativas a esta carpeta
    entrega3_root = Path(__file__).parent
    
    checks = [
        ("src/", entrega3_root / "src"),
        ("src/config.py", entrega3_root / "src" / "config.py"),
        ("src/preprocessing.py", entrega3_root / "src" / "preprocessing.py"),
        ("src/features.py", entrega3_root / "src" / "features.py"),
        ("src/models.py", entrega3_root / "src" / "models.py"),
        ("src/inference.py", entrega3_root / "src" / "inference.py"),
        ("src/app.py", entrega3_root / "src" / "app.py"),
    ]
    
    all_ok = True
    for name, path in checks:
        if path.exists():
            print(f"✅ {name}")
        else:
            print(f"❌ {name} no encontrado")
            all_ok = False
    
    return all_ok

def check_data():
    """Verifica si hay datos disponibles."""
    from src.config import POSES_DIR
    
    parquet_files = list(POSES_DIR.glob("*.parquet"))
    csv_files = list(POSES_DIR.glob("*.csv"))
    
    print(f"\n📊 Datos encontrados:")
    print(f"  Parquet files: {len(parquet_files)}")
    print(f"  CSV files: {len(csv_files)}")
    
    if len(parquet_files) > 0:
        print("✅ Archivos parquet encontrados")
    else:
        print("⚠️  No se encontraron archivos parquet")
        print("   Ejecuta process_videos.py para procesar videos")
    
    features_csv = POSES_DIR / "features_dataset.csv"
    if features_csv.exists():
        print("✅ Dataset de características encontrado")
        try:
            import pandas as pd
            df = pd.read_csv(features_csv)
            print(f"   Muestras: {len(df)}")
            if 'action' in df.columns:
                print(f"   Clases: {df['action'].nunique()}")
                print(f"   Distribución:")
                for cls, count in df['action'].value_counts().items():
                    print(f"     - {cls}: {count}")
        except ImportError:
            print("   ⚠️  No se puede leer el CSV (pandas no instalado)")
    else:
        print("⚠️  Dataset de características no encontrado")
        print("   Ejecuta train.py para generarlo")
    
    return len(parquet_files) > 0

def check_models():
    """Verifica si hay modelos entrenados."""
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("⚠️  Directorio models/ no existe")
        return False
    
    model_file = models_dir / "best_model.pkl"
    if model_file.exists():
        print("✅ Modelo entrenado encontrado")
        return True
    else:
        print("⚠️  Modelo no encontrado")
        print("   Ejecuta train.py para entrenar un modelo")
        return False

def main():
    print("=" * 50)
    print("Verificación del Entorno")
    print("=" * 50)
    
    print("\n1. Versión de Python:")
    py_ok = check_python_version()
    
    print("\n2. Dependencias:")
    deps_ok = check_dependencies()
    
    print("\n3. Estructura del proyecto:")
    struct_ok = check_structure()
    
    print("\n4. Datos:")
    data_ok = check_data()
    
    print("\n5. Modelos:")
    model_ok = check_models()
    
    print("\n" + "=" * 50)
    print("Resumen:")
    print("=" * 50)
    
    if py_ok and deps_ok and struct_ok:
        print("✅ Entorno básico: OK")
    else:
        print("❌ Entorno básico: FALTA CONFIGURAR")
        if not py_ok:
            print("   - Actualiza Python a 3.8+")
        if not deps_ok:
            print("   - Ejecuta: pip install -r requirements.txt")
        if not struct_ok:
            print("   - Verifica que todos los archivos estén presentes")
    
    if data_ok:
        print("✅ Datos: Disponibles")
    else:
        print("⚠️  Datos: Faltan archivos parquet")
    
    if model_ok:
        print("✅ Modelo: Entrenado y listo")
    else:
        print("⚠️  Modelo: Necesita entrenamiento")
    
    print("\n" + "=" * 50)
    
    if py_ok and deps_ok and struct_ok:
        if not data_ok:
            print("\n📝 Próximo paso: Ejecuta process_videos.py")
        elif not model_ok:
            print("\n📝 Próximo paso: Ejecuta train.py")
        else:
            print("\n🎉 ¡Todo listo! Ejecuta: python -m src.app")
    else:
        print("\n📝 Próximo paso: Configura el entorno básico primero")

if __name__ == "__main__":
    main()

