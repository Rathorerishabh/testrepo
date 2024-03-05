import mysql.connector
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd

def get_all_data(cursor):
    cursor.execute("SELECT gsr_value, heart_rate_value, temperature_value FROM sensor_3 ORDER BY timestamp_column DESC")
    result = cursor.fetchall()
    return result
gsr_universe = np.linspace(0, 10, 1000)
heart_rate_universe = np.linspace(50, 120, 1000)
temperature_universe = np.linspace(30, 45, 1000)
stress_level_universe = np.linspace(0, 1, 1000)

def create_control_system():
    # Define your fuzzy logic control system and rules
    gsr = ctrl.Antecedent(gsr_universe, 'gsr')
    gsr['low'] = fuzz.trapmf(gsr_universe, [0, 0, 1, 2])
    gsr['medium'] = fuzz.trapmf(gsr_universe, [1.5, 2, 4, 4.5])
    gsr['high'] = fuzz.trapmf(gsr_universe, [4, 4.5, 6, 6.5])
    gsr['very_high'] = fuzz.trapmf(gsr_universe, [6, 6.5, 10, 10])

    heart_rate = ctrl.Antecedent(heart_rate_universe, 'heart_rate')
    heart_rate['low'] = fuzz.trapmf(heart_rate_universe, [50, 50, 60, 70])
    heart_rate['medium'] = fuzz.trapmf(heart_rate_universe, [65, 70, 85, 90])
    heart_rate['high'] = fuzz.trapmf(heart_rate_universe, [85, 89, 96, 100])
    heart_rate['very_high'] = fuzz.trapmf(heart_rate_universe, [95, 100, 120, 120])

    temperature = ctrl.Antecedent(temperature_universe, 'temperature')
    temperature['low'] = fuzz.trapmf(temperature_universe, [30, 30, 32, 33])
    temperature['medium'] = fuzz.trapmf(temperature_universe, [32, 33, 35, 36])
    temperature['high'] = fuzz.trapmf(temperature_universe, [35, 36, 37, 38])
    temperature['very_high'] = fuzz.trapmf(temperature_universe, [37, 38, 45, 45])


    stress_level = ctrl.Consequent(stress_level_universe, 'stress_level')
    stress_level = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'stress_level')
    stress_level.automf(4, names=['relax', 'calm', 'anxious', 'stress'])

    rules = []

    rules.append(ctrl.Rule(gsr['low'] & temperature['low'] & heart_rate['low'], stress_level['relax']))
    rules.append(ctrl.Rule(gsr['low'] & temperature['low'] & heart_rate['medium'], stress_level['anxious']))
    rules.append(ctrl.Rule(gsr['low'] & temperature['low'] & heart_rate['high'], stress_level['anxious']))
    rules.append(ctrl.Rule(gsr['low'] & temperature['low'] & heart_rate['very_high'], stress_level['anxious']))

    rules.append(ctrl.Rule(gsr['low'] & temperature['medium'] & heart_rate['low'], stress_level['relax']))
    rules.append(ctrl.Rule(gsr['low'] & temperature['medium'] & heart_rate['medium'], stress_level['calm']))
    rules.append(ctrl.Rule(gsr['low'] & temperature['medium'] & heart_rate['high'], stress_level['anxious']))
    rules.append(ctrl.Rule(gsr['low'] & temperature['medium'] & heart_rate['very_high'], stress_level['anxious']))

    rules.append(ctrl.Rule(gsr['low'] & temperature['high'] & heart_rate['low'], stress_level['relax']))
    rules.append(ctrl.Rule(gsr['low'] & temperature['high'] & heart_rate['medium'], stress_level['calm']))
    rules.append(ctrl.Rule(gsr['low'] & temperature['high'] & heart_rate['high'], stress_level['calm']))
    rules.append(ctrl.Rule(gsr['low'] & temperature['high'] & heart_rate['very_high'], stress_level['anxious']))

    rules.append(ctrl.Rule(gsr['low'] & temperature['very_high'] & heart_rate['low'], stress_level['relax']))
    rules.append(ctrl.Rule(gsr['low'] & temperature['very_high'] & heart_rate['medium'], stress_level['relax']))
    rules.append(ctrl.Rule(gsr['low'] & temperature['very_high'] & heart_rate['high'], stress_level['calm']))
    rules.append(ctrl.Rule(gsr['low'] & temperature['very_high'] & heart_rate['very_high'], stress_level['anxious']))

# Medium GSR rules
    rules.append(ctrl.Rule(gsr['medium'] & temperature['low'] & heart_rate['low'], stress_level['anxious']))
    rules.append(ctrl.Rule(gsr['medium'] & temperature['low'] & heart_rate['medium'], stress_level['anxious']))
    rules.append(ctrl.Rule(gsr['medium'] & temperature['low'] & heart_rate['high'], stress_level['anxious']))
    rules.append(ctrl.Rule(gsr['medium'] & temperature['low'] & heart_rate['very_high'], stress_level['stress']))

    rules.append(ctrl.Rule(gsr['medium'] & temperature['medium'] & heart_rate['low'], stress_level['calm']))
    rules.append(ctrl.Rule(gsr['medium'] & temperature['medium'] & heart_rate['medium'], stress_level['calm']))
    rules.append(ctrl.Rule(gsr['medium'] & temperature['medium'] & heart_rate['high'], stress_level['anxious']))
    rules.append(ctrl.Rule(gsr['medium'] & temperature['medium'] & heart_rate['very_high'], stress_level['anxious']))

    rules.append(ctrl.Rule(gsr['medium'] & temperature['high'] & heart_rate['low'], stress_level['calm']))
    rules.append(ctrl.Rule(gsr['medium'] & temperature['high'] & heart_rate['medium'], stress_level['calm']))
    rules.append(ctrl.Rule(gsr['medium'] & temperature['high'] & heart_rate['high'], stress_level['calm']))
    rules.append(ctrl.Rule(gsr['medium'] & temperature['high'] & heart_rate['very_high'], stress_level['stress']))

    rules.append(ctrl.Rule(gsr['medium'] & temperature['very_high'] & heart_rate['low'], stress_level['relax']))
    rules.append(ctrl.Rule(gsr['medium'] & temperature['very_high'] & heart_rate['medium'], stress_level['calm']))
    rules.append(ctrl.Rule(gsr['medium'] & temperature['very_high'] & heart_rate['high'], stress_level['calm']))
    rules.append(ctrl.Rule(gsr['medium'] & temperature['very_high'] & heart_rate['very_high'], stress_level['anxious']))

# High GSR rules
    rules.append(ctrl.Rule(gsr['high'] & temperature['low'] & heart_rate['low'], stress_level['anxious']))
    rules.append(ctrl.Rule(gsr['high'] & temperature['low'] & heart_rate['medium'], stress_level['calm']))
    rules.append(ctrl.Rule(gsr['high'] & temperature['low'] & heart_rate['high'], stress_level['anxious']))
    rules.append(ctrl.Rule(gsr['high'] & temperature['low'] & heart_rate['very_high'], stress_level['anxious']))

    rules.append(ctrl.Rule(gsr['high'] & temperature['medium'] & heart_rate['low'], stress_level['anxious']))
    rules.append(ctrl.Rule(gsr['high'] & temperature['medium'] & heart_rate['medium'], stress_level['anxious']))
    rules.append(ctrl.Rule(gsr['high'] & temperature['medium'] & heart_rate['high'], stress_level['anxious']))
    rules.append(ctrl.Rule(gsr['high'] & temperature['medium'] & heart_rate['very_high'], stress_level['anxious']))

    rules.append(ctrl.Rule(gsr['high'] & temperature['high'] & heart_rate['low'], stress_level['calm']))
    rules.append(ctrl.Rule(gsr['high'] & temperature['high'] & heart_rate['medium'], stress_level['calm']))
    rules.append(ctrl.Rule(gsr['high'] & temperature['high'] & heart_rate['high'], stress_level['anxious']))
    rules.append(ctrl.Rule(gsr['high'] & temperature['high'] & heart_rate['very_high'], stress_level['stress']))

    rules.append(ctrl.Rule(gsr['high'] & temperature['very_high'] & heart_rate['low'], stress_level['calm']))
    rules.append(ctrl.Rule(gsr['high'] & temperature['very_high'] & heart_rate['medium'], stress_level['calm']))
    rules.append(ctrl.Rule(gsr['high'] & temperature['very_high'] & heart_rate['high'], stress_level['anxious']))
    rules.append(ctrl.Rule(gsr['high'] & temperature['very_high'] & heart_rate['very_high'], stress_level['stress']))

# Very High GSR rules
    rules.append(ctrl.Rule(gsr['very_high'] & temperature['low'] & heart_rate['low'], stress_level['anxious']))
    rules.append(ctrl.Rule(gsr['very_high'] & temperature['low'] & heart_rate['medium'], stress_level['anxious']))
    rules.append(ctrl.Rule(gsr['very_high'] & temperature['low'] & heart_rate['high'], stress_level['stress']))
    rules.append(ctrl.Rule(gsr['very_high'] & temperature['low'] & heart_rate['very_high'], stress_level['anxious']))

    rules.append(ctrl.Rule(gsr['very_high'] & temperature['medium'] & heart_rate['low'], stress_level['anxious']))
    rules.append(ctrl.Rule(gsr['very_high'] & temperature['medium'] & heart_rate['medium'], stress_level['anxious']))
    rules.append(ctrl.Rule(gsr['very_high'] & temperature['medium'] & heart_rate['high'], stress_level['anxious']))
    rules.append(ctrl.Rule(gsr['very_high'] & temperature['medium'] & heart_rate['very_high'], stress_level['stress']))

    rules.append(ctrl.Rule(gsr['very_high']& temperature['high'] & heart_rate['low'] , stress_level['anxious']))
    rules.append(ctrl.Rule(gsr['very_high']& temperature['high'] & heart_rate['medium'] , stress_level['calm']))
    rules.append(ctrl.Rule(gsr['very_high']&temperature['high'] & heart_rate['high']  , stress_level['anxious']))
    rules.append(ctrl.Rule(gsr['very_high']&temperature['high'] & heart_rate['very_high']  , stress_level['stress']))

    rules.append(ctrl.Rule(gsr['very_high']& temperature['very_high'] & heart_rate['low'] , stress_level['calm']))
    rules.append(ctrl.Rule(gsr['very_high']& temperature['very_high'] & heart_rate['medium'] , stress_level['calm']))
    rules.append(ctrl.Rule(gsr['very_high']&temperature['very_high'] & heart_rate['high']  , stress_level['anxious']))
    rules.append(ctrl.Rule(gsr['very_high']&temperature['very_high'] & heart_rate['very_high']  , stress_level['stress']))


    # ... (rest of your fuzzy rule definitions)

    ctrl_system = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(ctrl_system)
    
    return simulation
def classify_stress_level(gsr_value, heart_rate_value, temperature_value, simulation):
    simulation.input['gsr'] = gsr_value
    simulation.input['heart_rate'] = heart_rate_value
    simulation.input['temperature'] = temperature_value

    simulation.compute()

    stress_level = simulation.output['stress_level']
    return stress_level
def main():
    con = mysql.connector.connect(host='localhost', user='root', password='', database='sensoe1')
    cursor = con.cursor(dictionary=True)

    data_rows = get_all_data(cursor)
    simulation = create_control_system()

    results_list = []

    for data_row in data_rows:
        gsr_value = data_row['gsr_value']
        heart_rate_value = data_row['heart_rate_value']
        temperature_value = data_row['temperature_value']

        stress_level = classify_stress_level(gsr_value, heart_rate_value, temperature_value, simulation)

        relax_threshold = 0.20
        calm_threshold = 0.50
        anxious_threshold = 0.77

        if stress_level <= relax_threshold:
            classification = "Relaxed"
        elif relax_threshold < stress_level <= calm_threshold:
            classification = "Calm"
        elif calm_threshold < stress_level <= anxious_threshold:
            classification = "Anxious"
        else:
            classification = "Stressed"

        result_dict = {
            'gsr_value': gsr_value,
            'heart_rate_value': heart_rate_value,
            'temperature_value': temperature_value,
            'stress_level': stress_level,
            'classification': classification
        }

        results_list.append(result_dict)

    results_df = pd.DataFrame(results_list)
    print(results_df)

    # Specify the target table in your database
    target_table = 'your_table_name'

    # Insert the data into the specified table in the same database
    insert_data_into_database(results_df, target_table)

    cursor.close()
    con.close()

if __name__ == "__main__":
    main()
def view_all_rows(table_name):
    try:
        con = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='sensoe1'
        )
        cursor = con.cursor(dictionary=True)

        # Execute the SELECT query
        select_query = f"SELECT * FROM {table_name}"
        cursor.execute(select_query)

        # Fetch all rows
        rows = cursor.fetchall()

        # Print the rows
        for row in rows:
            print(row)

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        if con.is_connected():
            cursor.close()
            con.close()

# Usage
table_name_to_view = 'your_table_name'
view_all_rows(table_name_to_view)
