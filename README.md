# machine_learning_project

A small Python machine learning project. This README explains how to get a working development environment, run tests, and contribute.

Quick setup (recommended)

1. Clone the repository
    
    Fork the project on GitHub (optional) and clone your fork locally:
    ```
    git clone <your-repo-url>
    cd machine_learning_project
    ```

2. Create a virtual environment (optional, but recommended)

   For ease of use and ensure that there are no conflicts with any existing environments that you may have, it would be best to create a virtual environment from the root repository.
   
   Run the following on your terminal:
   ```
   python -m venv _ml_project_env
   source _ml_project_env/bin/activate
   ```

   Note: on your terminal, you can verify that it works if on left you see the following `(_ml_project_env)`. Make sure that this is the case in order to make sure that all packages will work as expected.

3. Install dependencies

    Run the command below in order to download all necessary packages:
    ```
    pip install -r requirements.txt
    pip install -e .
    ```

4. Run tests

    Run the test suite with pytest (from the root directory):
    ```
    pytest
    ```
    If you would like to run a specific test, you can also run the following
    ```
    pytest tests/{{file_name}}.py -q
    ```