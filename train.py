"""Compatibilidad con la antigua ruta; el pipeline vive en pipelines.train."""

from pipelines.train import main


if __name__ == "__main__":
    main()
