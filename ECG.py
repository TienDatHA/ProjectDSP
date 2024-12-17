import numpy as np
import PySimpleGUI as sg
from ECG_processing.ecg_lib import (
    read_csv,
    fit_peaks,
    check_peaks,
    update_rr,
    clean_rr_intervals,
)
from model_ulits import (
    load_and_preprocess_data,
    train_models,
    predict_heart_disease,
    format_prediction
)

def main_gui():
    sg.theme('LightBlue')
    layout = [
        [sg.Text("ECG Heart Rate Monitor", size=(30, 1), justification='center', font='Helvetica 20')],
        [sg.Button("Switch to Disease Prediction Mode", size=(30, 1), key='-SWITCH-')],
        [sg.Text("Select ECG CSV File:", justification='right'), sg.Input(key='-FILE-', enable_events=True, justification='right'), sg.FileBrowse(file_types=(("CSV Files", "*.csv"),))],
        [sg.Button("Process", size=(10, 1)), sg.Button("Exit", size=(10, 1))],
        [sg.Text("Results:", font='Helvetica 15')],
        [sg.Text("BPM:", size=(10, 1), justification='right'), sg.Text("", size=(10, 1), key='-BPM-')],
        [sg.Text("RR Intervals:", size=(10, 1), justification='right'), sg.Multiline("", size=(50, 5), key='-RR-', disabled=True, justification='right')],
        [sg.Text("Removed Beats:", size=(12, 1), justification='right'), sg.Multiline("", size=(50, 5), key='-REMOVED-', disabled=True, justification='right')],
    ]
    column_layout = [[sg.Column(layout, element_justification='right')]]
    window = sg.Window("Heart Rate Monitor", column_layout, size=(600, 400))

    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break
        if event == '-SWITCH-':
            window.close()
            disease_prediction_gui()
            break
        if event == "Process":
            file_path = values['-FILE-']
            if file_path:
                try:
                    # Process ECG data
                    hrdata = read_csv(file_path)
                    sample_rate = 1000  # Example sample rate, adjust as necessary
                    rol_mean = np.mean(hrdata)

                    # Initialize working_data dictionary
                    working_data = {}

                    # Fit peaks and process data
                    working_data = fit_peaks(hrdata, rol_mean, sample_rate, working_data=working_data)
                    rr_arr = working_data['RR_list']
                    peaklist = working_data['peaklist']
                    ybeat = working_data['ybeat']
                    working_data = check_peaks(rr_arr, peaklist, ybeat, working_data=working_data)
                    working_data = update_rr(working_data)
                    working_data = clean_rr_intervals(working_data)

                    # Calculate BPM
                    bpm = 60000 / np.mean(working_data['RR_list'])
                    if bpm < 60 or bpm > 100:
                        bpm_warning = ""
                    else:
                        bpm_warning = ""

                    # Update GUI
                    window['-BPM-'].update(f"{bpm:.2f}{bpm_warning}")
                    window['-RR-'].update(", ".join(f"{rr:.2f}" for rr in working_data['RR_list']))
                    window['-REMOVED-'].update(", ".join(f"{rb:.2f}" for rb in working_data['removed_beats']))

                except Exception as e:
                    sg.popup_error(f"Error processing ECG data: {str(e)}")

    window.close()

def disease_prediction_gui():
    layout = [
        [sg.Text('Enter Patient Information:', font=('Any', 14))],
        [sg.Text('Age (20-100):', justification='right'), sg.InputText(key='age', justification='right')],
        [sg.Text('Sex (1=male, 0=female):', justification='right'), sg.InputText(key='sex', justification='right')],
        [sg.Text('Chest Pain Type (0-3):', justification='right'), sg.InputText(key='cp', justification='right')],
        [sg.Text('Resting Blood Pressure (90-200 mm Hg):', justification='right'), sg.InputText(key='trestbps', justification='right')],
        [sg.Text('Cholesterol (100-600 mg/dl):', justification='right'), sg.InputText(key='chol', justification='right')],
        [sg.Text('Fasting Blood Sugar > 120 mg/dl (1=true, 0=false):', justification='right'), sg.InputText(key='fbs', justification='right')],
        [sg.Text('Resting ECG Results (0-2):', justification='right'), sg.InputText(key='restecg', justification='right')],
        [sg.Text('Maximum Heart Rate (60-220):', justification='right'), sg.InputText(key='thalach', justification='right')],
        [sg.Text('Exercise Induced Angina (1=yes, 0=no):', justification='right'), sg.InputText(key='exang', justification='right')],
        [sg.Text('ST Depression Induced by Exercise (0.0-6.0):', justification='right'), sg.InputText(key='oldpeak', justification='right')],
        [sg.Text('Slope of Peak Exercise ST Segment (0-2):', justification='right'), sg.InputText(key='slope', justification='right')],
        [sg.Text('Number of Major Vessels (0-3):', justification='right'), sg.InputText(key='ca', justification='right')],
        [sg.Text('Thalassemia (3=normal, 6=fixed defect, 7=reversible defect):', justification='right'), sg.InputText(key='thal', justification='right')],
        [sg.Button('Submit'), sg.Button('Back')],
        [sg.Text('Prediction Result:', font=('Any', 14))],
        [sg.Multiline(size=(50, 10), key='output', disabled=True)]
    ]
    column_layout = [[sg.Column(layout, element_justification='right')]]
    window = sg.Window('Heart Disease Prediction', column_layout)

    # Load data and train models
    X_train, X_test, y_train, y_test = load_and_preprocess_data(r"D:\\Study\\code\\Python\\ProjectDSP\\heart.csv")
    models = train_models(X_train, y_train)
    best_model = models["Logistic Regression"]  # Assuming Random Forest is the best model

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == 'Back':
            window.close()
            main_gui()
            break
        if event == 'Submit':
            try:
                patient_data = {
                    'age': int(values['age']),
                    'sex': int(values['sex']),
                    'cp': int(values['cp']),
                    'trestbps': int(values['trestbps']),
                    'chol': int(values['chol']),
                    'fbs': int(values['fbs']),
                    'restecg': int(values['restecg']),
                    'thalach': int(values['thalach']),
                    'exang': int(values['exang']),
                    'oldpeak': float(values['oldpeak']),
                    'slope': int(values['slope']),
                    'ca': int(values['ca']),
                    'thal': int(values['thal'])
                }
                print("Patient Data:", patient_data)  # Debugging statement
                prob, pred = predict_heart_disease(best_model, patient_data)
                result = format_prediction(prob, pred)
                output_text = "\n".join([f"{key}: {value}" for key, value in result.items()])
                print("Prediction Result:", output_text)  # Debugging statement
                window['output'].update(output_text)
            except ValueError as e:
                window['output'].update(f"Error: {str(e)}")
            except Exception as e:
                window['output'].update(f"Unexpected Error: {str(e)}")

    window.close()

if __name__ == "__main__":
    main_gui()