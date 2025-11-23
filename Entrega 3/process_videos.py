"""
Script para procesar videos y extraer landmarks y características extendidas.
Útil si necesitas procesar videos desde Google Drive o localmente.
"""
import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.config import PROJECT_ROOT, POSES_DIR, VIDEOS_DIR
from src.preprocessing import extract_poses_from_video


def process_videos_from_directory(
    videos_dir: str,
    output_dir: str = None,
    stride: int = 1
):
    """
    Procesa todos los videos en un directorio y guarda los landmarks y características.

    Args:
        videos_dir: Directorio con videos.
        output_dir: Directorio de salida (default: POSES_DIR).
        stride: Procesar cada N frames.
    """
    if output_dir is None:
        output_dir = POSES_DIR
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = Path(videos_dir)

    # Buscar videos
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(videos_dir.rglob(f'*{ext}'))

    print(f"Encontrados {len(video_files)} videos")

    summaries = []

    for video_path in tqdm(video_files, desc="Procesando videos"):
        try:
            # Crear nombre de salida basado en la estructura de carpetas
            rel_path = video_path.relative_to(videos_dir)
            # Si está en una subcarpeta (ej: subject1/video.mp4)
            if len(rel_path.parts) > 1:
                subject = rel_path.parts[0]
                video_name = rel_path.stem
                output_name = f"{subject}__{video_name}__{video_name}.parquet"
            else:
                output_name = f"{video_path.stem}__{video_path.stem}.parquet"

            output_path = output_dir / output_name

            # Extraer poses con características extendidas
            df = extract_poses_from_video(str(video_path), stride=stride)

            if len(df) > 0:
                # Guardar parquet con landmarks y features
                df.to_parquet(output_path, index=False)

                # Resumen
                summary = {
                    'video': str(video_path),
                    'parquet': output_name,
                    'frames': len(df),
                    'fps': df['fps'].iloc[0] if 'fps' in df.columns else 0,
                    'width': df['width'].iloc[0] if 'width' in df.columns else 0,
                    'height': df['height'].iloc[0] if 'height' in df.columns else 0,
                }

                # Detecciones válidas
                x_cols = [f'x_{i}' for i in range(33)]
                valid_frames = (~df[x_cols].isna().all(axis=1)).sum()
                summary['valid_frames'] = int(valid_frames)
                summary['valid_ratio'] = valid_frames / len(df) if len(df) > 0 else 0
                
                summaries.append(summary)
                print(f"✓ {video_path.name}: {len(df)} frames, {valid_frames} válidos")
            else:
                print(f"✗ {video_path.name}: Sin frames válidos")

        except Exception as e:
            print(f"✗ Error procesando {video_path.name}: {e}")
            continue

    # Guardar resumen
    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = output_dir / "poses_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nResumen guardado en: {summary_path}")
        print(f"\nTotal procesado: {len(summaries)} videos")
        print(f"Frames totales: {summary_df['frames'].sum()}")
        print(f"Frames válidos: {summary_df['valid_frames'].sum()}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Procesa videos y extrae landmarks usando MediaPipe con features extendidas"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(VIDEOS_DIR),
        help="Directorio con videos a procesar"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directorio de salida para archivos parquet (default: Entrega 2/data/poses)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Procesar cada N frames (default: 1 = todos)"
    )

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Directorio no encontrado: {args.input}")
        sys.exit(1)

    process_videos_from_directory(
        videos_dir=args.input,
        output_dir=args.output,
        stride=args.stride
    )


if __name__ == "__main__":
    main()
