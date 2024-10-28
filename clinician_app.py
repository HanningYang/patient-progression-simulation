import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gspread
import math
from scipy.stats import lognorm
from oauth2client.service_account import ServiceAccountCredentials

params1 = []
if 'parameters_saved' not in st.session_state:
    st.session_state['parameters_saved'] = False

def save_parameters(name, comment, parameters, add_variability, variability_level, birth_means_var):
    try:
        # Use credentials from Streamlit secrets
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(
            st.secrets["google_service_account"],
            scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        )
        client = gspread.authorize(credentials)

        # Access the sheet
        sheet = client.open("ODE-parameters").sheet1
        
        # Prepare the data to append
        data = [name, comment] + parameters + [add_variability, variability_level] + birth_means_var
        
        # Append the data as a new row
        sheet.append_row(data)
    except gspread.exceptions.APIError as e:
        st.error(f"API Error: {e}")





### Function Definitions

def ode_system1(y, t, K_C, K_H, K_W, K_A, K_I, r_C, r_H, r_W, r_A, r_I,
                alpha_CW, alpha_HW, alpha_AW, alpha_IW, beta_HC, theta_HI):
    C, H, W, A, I = y
    dCdt = r_C * C * (1 - C / K_C) + alpha_CW * (W / K_W)
    dHdt = r_H * H * (1 - H / K_H) + alpha_HW * (W / K_W) + beta_HC * (C / K_C) + theta_HI * (I / K_I)
    dWdt = r_W * W * (1 - W / K_W)
    dAdt = r_A * A * (1 - A / K_A) + alpha_AW * (W / K_W)
    dIdt = r_I * I * (1 - I / K_I) + alpha_IW * (W / K_W)
    return np.array([dCdt, dHdt, dWdt, dAdt, dIdt])

def ode_system2(y, t, K_C, K_H, K_W, K_A, K_I, r_C, r_H, r_W, r_A, r_I,
                alpha_CW, alpha_HW, alpha_AW, alpha_IW, beta_HC, theta_HI, delta):
    C, H, W, A, I = y
    dCdt = (r_C + delta) * C * (1 - C / K_C) + alpha_CW * (W / K_W)
    dHdt = r_H * H * (1 - H / K_H) + alpha_HW * (W / K_W) + beta_HC * (C / K_C) + theta_HI * (I / K_I)
    dWdt = r_W * W * (1 - W / K_W)
    dAdt = r_A * A * (1 - A / K_A) + alpha_AW * (W / K_W)
    dIdt = r_I * I * (1 - I / K_I) + alpha_IW * (W / K_W)
    return np.array([dCdt, dHdt, dWdt, dAdt, dIdt])

def euler_method(func, y0, t, *args):
    ys = [y0]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        y_prev = ys[-1]
        dydt = func(y_prev, t[i - 1], *args)
        y_new = y_prev + dydt * dt
        ys.append(y_new)
    solu = np.array(ys)
    return solu

def apply_noise(data, size, noise_std):
    noise_C = np.random.normal(0, noise_std[0], size)
    noise_H = np.random.normal(0, noise_std[1], size)
    noise_W = np.random.normal(0, noise_std[2], size)
    noise_A = np.random.normal(0, noise_std[3], size)
    noise_I = np.random.normal(0, noise_std[4], size)
    noise = np.stack([noise_C, noise_H, noise_W, noise_A, noise_I], axis=1)
    data_with_noise = data + noise
    # Adjust noise to ensure no negative values in data_with_noise
    negative_indices = data_with_noise < 0
    abs_noise = np.abs(noise)
    data_with_noise[negative_indices] = data_with_noise[negative_indices] - noise[negative_indices] + abs_noise[negative_indices]

    return data_with_noise

def process_data(data, size, noise, noise_std):
    if noise:
        data = apply_noise(data.copy(), size, noise_std)
    return data

def concatenate_data_diff_noise(init_inter, init_se, time_points, maxes, cali_params, noise_std, noise, type='hypo', inter_num=None):
    if type == 'real':
        if inter_num is None:
            raise ValueError("inter_num must be provided when type is 'real'")
        inter_time_points = time_points[:inter_num]
        se_time_points = time_points[inter_num:]        
    else:
        mid_index = len(time_points) // 2
        inter_time_points = time_points[:mid_index]
        se_time_points = time_points[mid_index:]

    final_data = []
    K_C, K_H, K_W, K_A, K_I = maxes
    r_C, r_H, r_W, r_A, r_I, alpha_CW, alpha_HW, alpha_AW, alpha_IW, beta_HC, theta_HI, delta = cali_params

    for idx, t in enumerate(inter_time_points):
        t = np.array(t)
        y0 = init_inter[:, idx]
        inter = euler_method(ode_system1, y0, t, K_C, K_H, K_W, K_A, K_I,
                             r_C, r_H, r_W, r_A, r_I,
                             alpha_CW, alpha_HW, alpha_AW, alpha_IW,
                             beta_HC, theta_HI)
        size = inter.shape[0]
        inter = process_data(inter, size, noise, noise_std)
        t = t.reshape(-1, 1)
        inter = np.hstack([inter, t])
        final_data.append(inter)

    for idx, t in enumerate(se_time_points):
        t = np.array(t)
        y0 = init_se[:, idx]
        se = euler_method(ode_system2, y0, t, K_C, K_H, K_W, K_A, K_I,
                          r_C, r_H, r_W, r_A, r_I,
                          alpha_CW, alpha_HW, alpha_AW, alpha_IW,
                          beta_HC, theta_HI, delta)
        size = se.shape[0]
        se = process_data(se, size, noise, noise_std)
        t = t.reshape(-1, 1)
        se = np.hstack([se, t])
        final_data.append(se)

    return final_data

def resample_positive(mean, std, size):
    values = np.random.normal(loc=mean, scale=std, size=size)
    while (values < 0).any():
        negative_indices = values < 0
        values[negative_indices] = np.random.normal(loc=mean, scale=std, size=negative_indices.sum())
    return values

def generate_initial_conditions(num_patient, mean_C, std_C, mean_H, mean_W, mean_A, mean_I):
    # Generate initial conditions based on the means provided
    # sigma = 1.3  # Standard deviation for C cells
    # mu_simu_prime = np.log(mu_simu**2 / np.sqrt(mu_simu**2 + sigma**2))
    # sigma_simu_prime = np.sqrt(np.log(1 + (sigma**2) / mu_simu**2))
    # C_simu = np.exp(np.random.normal(loc=mu_simu_prime, scale=sigma_simu_prime, size=(num_patient,)))

    sigma_sq = math.log(1 + (std_C / mean_C)**2)
    mu = math.log(mean_C) - (sigma_sq / 2)
    sigma = math.sqrt(sigma_sq)
    C_simu = lognorm.rvs(s=sigma, scale=np.exp(mu), size=num_patient)
    
    H_simu = resample_positive(mean=mean_H, std=2.5, size=(num_patient,))
    W_simu = resample_positive(mean=mean_W, std=2.3, size=(num_patient,))
    A_simu = resample_positive(mean=mean_A, std=0.9, size=(num_patient,))
    I_simu = resample_positive(mean=mean_I, std=21, size=(num_patient,))
    
    simu_init = np.stack([C_simu, H_simu, W_simu, A_simu, I_simu])
    return simu_init


def plot_simulation(final_data):
    # Define colors for each biomarker
    colors = ['royalblue', 'mediumseagreen', 'salmon', 'gold', 'plum']
    # y_limits = [(0, 200), (0, 20), (5, 30), (0.0, 8.0), (0, 150)]
    biomarkers = ['CRP', 'Haemoglobin', 'BMI', 'Albumin', 'Iron']

    # Plot each patient’s data in separate figures
    num_patients = len(final_data)
    for idx, patient_data in enumerate(final_data):
        fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(10, 30), sharex=True)
        time = patient_data[:, -1]
        
        # Plot each biomarker separately
        for i, ax in enumerate(axes[:5]):
            ax.plot(time, patient_data[:, i], label=biomarkers[i], color=colors[i])
            # ax.set_ylim(y_limits[i])
            ax.set_ylabel(biomarkers[i])
            ax.legend()
        
        # Combined plot for all biomarkers in the last subplot
        for i, biomarker in enumerate(biomarkers):
            axes[5].plot(time, patient_data[:, i], label=biomarker, color=colors[i])
        axes[5].set_ylabel('Values')
        axes[5].set_title('Combined Biomarkers')
        axes[5].legend()
        
        # Set labels and titles
        axes[-1].set_xlabel('Time')
        if idx < num_patients // 2:
            fig.suptitle(f'Intermediate Patient {idx + 1} Progression')
        else:
            fig.suptitle(f'Severe Patient {idx + 1 - num_patients // 2} Progression')

        fig.tight_layout(rect=[0, 0.02, 1, 0.98])
        st.pyplot(fig)


# Set plot style
# plt.style.use('seaborn-darkgrid')
plt.style.use('ggplot')


st.title("RDEB Patient Progression Simulation")

st.markdown("""
Welcome to **RDEB Patient Progression Simulation App**. This app allows you to adjust simulation parameters and visualize the impact on RDEB patient outcomes.

**Instructions:**
- Adjust the parameters in the sidebar, where each parameter includes a brief description. The default values serve as initial estimates.
- Click **"Run Simulation"** to update the results.
- Click **"Save Parameters"** to store the parameter settings you find reasonable.
""")

# Sidebar parameters
st.sidebar.title("Adjust Parameters")

# st.sidebar.markdown("### Simulation Parameters")


# Explanation for parameters
# st.sidebar.info("""
# # **Parameters:**

# - **Growth Rates (r_C, r_H, r_W, r_A, r_I)**: Control the growth rate of each cell type.
# - **Interaction Parameters (alpha, beta, theta)**: Define how different biomarkers influence each other.
# - **Noise Level**: Adjusts the variability in the simulation data.
# - **Initial Conditions**: Mean values for the initial quantities of each cell type.
# """)

# Add a brief description
st.sidebar.markdown(""" Please enter the average values for each biomarker measured at birth for the intermediate and severe patient group. For CRP, also include the variability (standard deviation) to assess its range. """)


# Initial conditions
# Intermediate Group inputs
st.sidebar.markdown("### Intermediate Group")
st.sidebar.markdown("#### Mean of Biomarker")
mean_H_simu_inter = st.sidebar.number_input('Intermediate Haemoglobin', value=10.6)
mean_W_simu_inter = st.sidebar.number_input('Intermediate BMI', value=14.0)
mean_A_simu_inter = st.sidebar.number_input('Intermediate Albumin', value=3.8)
mean_I_simu_inter = st.sidebar.number_input('Intermediate Iron', value=32.5)
mean_C_simu_inter = st.sidebar.number_input('Intermediate CRP', value=19.0)

st.sidebar.markdown("#### Variability of CRP")
std_C_inter = st.sidebar.number_input('Intermediate CRP Variability', value=29.0)

# Severe Group inputs
st.sidebar.markdown("### Severe Group")
st.sidebar.markdown("#### Mean of Biomarker")
mean_H_simu_se = st.sidebar.number_input('Severe Haemoglobin', value=8.4)
mean_W_simu_se = st.sidebar.number_input('Severe BMI', value=14.0)
mean_A_simu_se = st.sidebar.number_input('Severe Albumin', value=2.8)
mean_I_simu_se = st.sidebar.number_input('Severe Iron', value=16.4)
mean_C_simu_se = st.sidebar.number_input('Severe CRP', value=74.0)

st.sidebar.markdown("#### Variability of CRP")
std_C_se = st.sidebar.number_input('Severe CRP Variability', value=55.0)



# Parameters to adjust
st.sidebar.markdown("### Growth/Deacay Rate of Different Biomarkers")
st.sidebar.markdown("#### A positive value for the growth/decay rate indicates that the biomarker tends to increase (growth) over time, while a negative value indicates a decrease (decay) over time.")
r_C1 = st.sidebar.slider('CRP', min_value=-1.0, max_value=1.0, value=0.04, step=0.01, format="%.2f")
r_H1 = st.sidebar.slider('Haemoglobin', min_value=-1.0, max_value=1.0, value=-0.12, step=0.01, format="%.2f")
r_W1 = st.sidebar.slider('BMI', min_value=-1.0, max_value=1.0, value=-0.05, step=0.01, format="%.2f")
r_A1 = st.sidebar.slider('Albumin', min_value=-1.0, max_value=1.0, value=-0.05, step=0.01, format="%.2f")
r_I1 = st.sidebar.slider('Iron', min_value=-1.0, max_value=1.0, value=-0.14, step=0.01, format="%.2f")

st.sidebar.markdown("### Difference in Growth/Decay Rate for Severe Group Relative to Intermediate Group to Differentiate Patient Severity")
st.sidebar.markdown("#### A positive correlation means that as one variable increases or decreases, the other tends to change in the same direction. A negative correlation means that as one variable increases, the other tends to decrease, and vice versa. The larger the absolute value, the stronger the correlation.")

delta1 = st.sidebar.slider('CRP Growth/Decay Difference', min_value=-1.0, max_value=1.0, value=0.16, step=0.01, format="%.2f")

st.sidebar.markdown("### How Different Biomarkers Influence Each Other")
alpha_CW1 = st.sidebar.slider('CRP and BMI', min_value=-1.0, max_value=1.0, value=0.16, step=0.01, format="%.2f")
alpha_HW1 = st.sidebar.slider('Haemoglobin and BMI', min_value=-1.0, max_value=1.0, value=-0.12, step=0.01, format="%.2f")
alpha_AW1 = st.sidebar.slider('Albumin and BMI', min_value=-1.0, max_value=1.0, value=-0.14, step=0.01, format="%.2f")
alpha_IW1 = st.sidebar.slider('Iron and BMI', min_value=-1.0, max_value=1.0, value=0.07, step=0.01, format="%.2f")
beta_HC1 = st.sidebar.slider('Haemoglobin and CRP', min_value=-1.0, max_value=1.0, value=0.16, step=0.01, format="%.2f")
theta_HI1 = st.sidebar.slider('Haemoglobin and Iron', min_value=-1.0, max_value=1.0, value=-0.06, step=0.01, format="%.2f")


st.sidebar.markdown("### Variability")
st.sidebar.markdown("#### If you choose not to add variability, you can leave the variability level alone.")
add_noise = st.sidebar.checkbox('Add Variability', True)
noise_level = st.sidebar.number_input('Variability Level', min_value=0.0, max_value=2.0, value=0.5, step=0.1, format="%.1f")


# Run simulation button
if st.sidebar.button('Run Simulation'):
    # Convert parameters to the appropriate format
    params1 = [r_C1, r_H1, r_W1, r_A1, r_I1,
               alpha_CW1, alpha_HW1, alpha_AW1, alpha_IW1,
               beta_HC1, theta_HI1, delta1]
    st.session_state['params1'] = params1

    
    # Adjust noise level
    noise_C_std = noise_level * 40
    noise_H_std = noise_level * 2.5  # std_H
    noise_W_std = noise_level * 2.3  # std_W
    noise_A_std = noise_level * 0.9  # std_A
    noise_I_std = noise_level * 21   # std_I
    noise_std = [noise_C_std, noise_H_std, noise_W_std, noise_A_std, noise_I_std]

    # Generate initial conditions based on the input means
    num_patient = 3
    simu_init_inter = generate_initial_conditions(
        num_patient,
        mean_C_simu_inter,
        std_C_inter,
        mean_H_simu_inter,
        mean_W_simu_inter,
        mean_A_simu_inter,
        mean_I_simu_inter
    )
    simu_init_se = generate_initial_conditions(
        num_patient,
        mean_C_simu_se,
        std_C_se,
        mean_H_simu_se,
        mean_W_simu_se,
        mean_A_simu_se,
        mean_I_simu_se
    )
    
    # Time points
    num_simu_patient = 2 * num_patient
    time_points_simu = [[i for i in range(16)] for _ in range(num_simu_patient)]
    
    # Maxes
    K_C = 200.0
    K_H = 15.0
    K_W = 25.0
    K_A = 5.0
    K_I = 160.0
    maxes = K_C, K_H, K_W, K_A, K_I

    # Run the simulation
    final_data = concatenate_data_diff_noise(
        simu_init_inter, simu_init_se, time_points_simu,
        maxes, params1, noise_std, noise=add_noise, type='hypo'
    )

    # Check for zero values in final_data
    contains_zero = any((patient_data[:, :-1] <= 0).any() for patient_data in final_data)
    if contains_zero:
        st.warning("The generated data contains zero or negative values. Please review your parameters or initial conditions.")
    else:
        st.success("No zero or negative values generated")


    
    # Visualize the results
    st.markdown("## Simulation Results")
    plot_simulation(final_data)
else:
    st.markdown("## Awaiting Simulation")
    st.markdown("Adjust parameters and click **Run Simulation** to see results.")


# Collect information from the UI elements and session_state
add_variability_state = 'T' if add_noise else 'F'
variability_level_value = noise_level if add_noise else "N/A"

# Collect mean values at birth for intermediate and severe groups
birth_means_var = [
    mean_H_simu_inter, mean_W_simu_inter, mean_A_simu_inter, mean_I_simu_inter, mean_C_simu_inter, std_C_inter,
    mean_H_simu_se, mean_W_simu_se, mean_A_simu_se, mean_I_simu_se, mean_C_simu_se, std_C_se
]

# Save parameters section
st.sidebar.markdown("### Save Your Parameters")
user_name = st.sidebar.text_input("Your Name")
user_comment = st.sidebar.text_area("Comments (Why did you choose these parameters?)")

# Ensure params1 exists in session state before accessing
if st.sidebar.button('Save Parameters'):
    if not user_name:
        st.warning("Please enter your name before saving.")
    elif 'params1' not in st.session_state:
        st.warning("Please run the simulation first to generate parameters.")
    else:
        save_parameters(
            user_name, 
            user_comment, 
            [float(p) for p in st.session_state['params1']],
            add_variability_state, 
            variability_level_value,
            birth_means_var
        )
        st.session_state['parameters_saved'] = True  # Set flag to True after saving

# Display the success message only once
if st.session_state['parameters_saved']:
    st.success("Parameters saved successfully!")
    



