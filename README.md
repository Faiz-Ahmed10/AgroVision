# Agrovision Project

This project uses Python 11 and manages dependencies through Miniconda. Follow the steps below to set up and run the project.

## Prerequisites

- Git
- Miniconda (Download from [Miniconda Official Website](https://docs.conda.io/en/latest/miniconda.html))

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd AgroVision
   ```

2. **Create Conda Environment**
   ```bash
   conda create -n agrovision python=11
   conda activate agrovision
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

1. **Activate the Environment**
   ```bash
   conda activate agrovision
   ```

2. **Run the Project**
   ```bash
   python app.py
   ```

## Troubleshooting

### Common Issues

1. **Python Version Mismatch**
   - If you encounter Python version issues, verify your conda environment:
     ```bash
     python --version
     ```
   - If needed, recreate the environment with the correct Python version.

2. **Package Installation Errors**
   - If pip installation fails, try updating pip:
     ```bash
     pip install --upgrade pip
     ```
   - Then retry installing requirements.

### Environment Management

- To deactivate the conda environment:
  ```bash
  conda deactivate
  ```

- To remove the environment if needed:
  ```bash
  conda env remove -n agrovision
  ```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

[Add your contact information here]
