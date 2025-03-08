# Module 5: Mini Project-1 Part-B  
## Bike Rental Prediction  
### Testing and Packaging  

For this project, we will test and package a bike rental count prediction system using modular programming. Please refer to Module 5 - AST 2 for this mini-project.

---

### **PART B [Mini-project Session - 8th March 2025]**

#### **Step 1: Ensure to go through the previous mini-project [Part A]**

---

#### **Step 2: Project Setup in VS Code** (2 points)
1. Use the existing project folder from the previous mini-project Part A session and open it in VS Code.
2. Update the project structure by creating new files for testing and packaging, as shown below:
   - **Files for testing**:
     - `conftest.py`
     - `test_features.py`
     - `test_predictions.py`
   - **Test requirements**
   - **Files related to packaging**:
     - `pyproject.toml`
     - `setup.py`
     - `manifest.in`
     - `mypy.ini`

---

#### **Step 3: Implement the following test cases** (3 points)
Implement test cases for:
- Pipeline processing steps, including:
  - Imputation
  - Mapping
  - Custom class transformations
- Prediction steps

---

#### **Step 4: Create a Virtual Environment**
1. Open the terminal in VS Code and navigate to the project folder.
2. Create a virtual environment as demonstrated in Module 5 - AST 1.

---

#### **Step 5: Install Dependencies** (1 point)
1. Activate the virtual environment in the terminal.
2. Install the necessary dependencies by running the `pip install` command for required libraries.

---

#### **Step 6: Train the Model** (1 point)
1. Execute the `train_pipeline.py` script to train the bike rental prediction model using the prepared data.

---

#### **Step 7: Run Test Cases** (2 points)
1. Run the test cases (created in Step 3) by executing the `pytest` command in the terminal.

---

#### **Step 8: Build a Package for the Application** (1 point)
1. Install the `build` library by running the `pip install` command.
2. Run the `build` command to create distributable files (`.tar`, `.whl`, etc.).

---
