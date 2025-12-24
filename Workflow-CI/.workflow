name: Continuous Integration (Skilled)

on: [push]

jobs:
  build_and_train:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install Dependencies
        run: pip install mlflow pandas scikit-learn

      - name: Run MLflow Project
        # Menjalankan project yang ada di dalam folder MLProject
        run: |
          mlflow run Workflow-CI/MLProject --no-conda --param-list n_estimators=50 max_depth=5

      - name: Upload Model Artifact
        uses: actions/upload-artifact@v4
        with:
          name: model-hasil-training
          path: mlruns/  # Mengupload hasil run MLflow sebagai bukti