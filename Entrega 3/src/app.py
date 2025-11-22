"""
Aplicación principal para detección en tiempo real
"""
import cv2
import argparse
import sys
from pathlib import Path

from src.inference import RealTimeActivityDetector
from src.config import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(
        description="Sistema de detección de actividades humanas en tiempo real"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(PROJECT_ROOT / "models" / "best_model.pkl"),
        help="Ruta al modelo entrenado"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Índice de la cámara (default: 0)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Ruta a video para procesar (opcional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta para guardar video procesado (opcional)"
    )
    
    args = parser.parse_args()
    
    # Verificar que el modelo existe
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Modelo no encontrado en {model_path}")
        print("Por favor, entrena un modelo primero usando train.py")
        sys.exit(1)
    
    # Inicializar detector
    print("Cargando modelo...")
    detector = RealTimeActivityDetector(str(model_path))
    
    # Abrir video o cámara
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video {args.video}")
            sys.exit(1)
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir la cámara {args.camera}")
            sys.exit(1)
    
    # Configurar video de salida si se especifica
    writer = None
    if args.output:
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print("\n=== Sistema de Detección de Actividades ===")
    print("Presiona 'q' para salir")
    print("Presiona 'r' para reiniciar el buffer")
    print("Presiona 's' para guardar screenshot\n")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    print("Fin del video")
                    break
                else:
                    continue
            
            # Procesar frame
            frame_annotated, activity, confidence = detector.process_frame(frame)
            
            # Mostrar información adicional
            info_text = f"Frame: {frame_count} | Buffer: {len(detector.frame_buffer)}/{detector.window_size}"
            cv2.putText(frame_annotated, info_text, (10, frame_annotated.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Guardar frame si hay writer
            if writer:
                writer.write(frame_annotated)
            
            # Mostrar frame
            cv2.imshow("Detección de Actividades", frame_annotated)
            
            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                detector.reset()
                print("Buffer reiniciado")
            elif key == ord('s'):
                screenshot_path = f"screenshot_{frame_count}.jpg"
                cv2.imwrite(screenshot_path, frame_annotated)
                print(f"Screenshot guardado: {screenshot_path}")
            
            frame_count += 1
            
            # Mostrar predicción en consola cada 30 frames
            if activity and frame_count % 30 == 0:
                print(f"Frame {frame_count}: {activity} (confianza: {confidence:.2%})")
    
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Recursos liberados")


if __name__ == "__main__":
    main()

