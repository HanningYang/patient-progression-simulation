import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### Function Definitions

def ode_system1(y, t, K_C, K_H, K_W, K_A, K_I, r_C, r_H, r_W, r_A, r_I,
                alpha_CW, alpha_HW, alpha_AW, alpha_IW, beta_HC, theta_HI):
    C, H, W, A, I = y
    dCdt = r_C * C * (1 - C / K_C) - alpha_CW * (W / K_W)
    dHdt = r_H * H * (1 - H / K_H) + alpha_HW * (W / K_W) - beta_HC * (C / K_C) + theta_HI * (I / K_I)
    dWdt = r_W * W * (1 - W / K_W)
    dAdt = r_A * A * (1 - A / K_A) + alpha_AW * (W / K_W)
    dIdt = r_I * I * (1 - I / K_I) + alpha_IW * (W / K_W)
    return np.array([dCdt, dHdt, dWdt, dAdt, dIdt])

def ode_system2(y, t, K_C, K_H, K_W, K_A, K_I, r_C, r_H, r_W, r_A, r_I,
                alpha_CW, alpha_HW, alpha_AW, alpha_IW, beta_HC, theta_HI, delta):
    C, H, W, A, I = y
    dCdt = (r_C + delta) * C * (1 - C / K_C) - alpha_CW * (W / K_W)
    dHdt = r_H * H * (1 - H / K_H) + alpha_HW * (W / K_W) - beta_HC * (C / K_C) + theta_HI * (I / K_I)
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
    # Adjust negative values
    data_with_noise[data_with_noise < 0] = 0
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

def generate_initial_conditions(num_patient, mu_simu, mean_H, mean_W, mean_A, mean_I):
    # Generate initial conditions based on the means provided
    sigma = 1.3  # Standard deviation for C cells
    mu_simu_prime = np.log(mu_simu**2 / np.sqrt(mu_simu**2 + sigma**2))
    sigma_simu_prime = np.sqrt(np.log(1 + (sigma**2) / mu_simu**2))
    C_simu = np.exp(np.random.normal(loc=mu_simu_prime, scale=sigma_simu_prime, size=(num_patient,)))
    
    H_simu = resample_positive(mean=mean_H, std=2.5, size=(num_patient,))
    W_simu = resample_positive(mean=mean_W, std=2.3, size=(num_patient,))
    A_simu = resample_positive(mean=mean_A, std=0.9, size=(num_patient,))
    I_simu = resample_positive(mean=mean_I, std=21, size=(num_patient,))
    
    simu_init = np.stack([C_simu, H_simu, W_simu, A_simu, I_simu])
    return simu_init

def plot_simulation(final_data):
    num_patients = len(final_data)
    for idx, patient_data in enumerate(final_data):
        fig, ax = plt.subplots(figsize=(10, 6))
        time = patient_data[:, -1]
        for i, label in enumerate(['C Cells', 'H Cells', 'W Cells', 'A Cells', 'I Cells']):
            ax.plot(time, patient_data[:, i], label=label)
        ax.set_title(f'Patient {idx + 1} Progression')
        ax.set_xlabel('Time')
        ax.set_ylabel('Cell Counts')
        ax.legend()
        st.pyplot(fig)


# Set plot style
# plt.style.use('seaborn-darkgrid')
plt.style.use('ggplot')


st.title("RDEB Patient Progression Simulation")

st.markdown("""
Welcome to the **RDEB Patient Progression Simulation App**. This app allows you to adjust simulation parameters and visualize the impact on RDEB patient outcomes.

**Instructions:**
- Adjust the parameters in the sidebar.
- Each parameter includes a brief description.
- Click **"Run Simulation"** to update the results.
""")

# Sidebar parameters
st.sidebar.title("Adjust Parameters")

st.sidebar.markdown("### Simulation Parameters")

# Explanation for parameters
st.sidebar.info("""
**Parameters:**

- **Growth Rates (r_C, r_H, r_W, r_A, r_I)**: Control the growth rate of each cell type.
- **Interaction Parameters (alpha, beta, theta)**: Define how different biomarkers influence each other.
- **Noise Level**: Adjusts the variability in the simulation data.
- **Initial Conditions**: Mean values for the initial quantities of each cell type.
""")

# Parameters to adjust
r_C1 = st.sidebar.slider('Growth/Deacay Rate of CRP', 0.0, 1.0, 0.18, 0.01)
r_H1 = st.sidebar.slider('Growth/Deacay Rate of haemoglobin)', -0.1, 0.1, 0.05, 0.01)
r_W1 = st.sidebar.slider('Growth/Deacay Rate of BMI', 0.0, 1.0, 0.1, 0.01)
r_A1 = st.sidebar.slider('Growth/Deacay Rate of albumin', 0.0, 0.1, 0.01, 0.005)
r_I1 = st.sidebar.slider('Growth/Deacay Rate of iron', -1.0, 0.0, -0.2, 0.05)

alpha_CW1 = st.sidebar.slider('correlation between CRP and BMI', 0.0, 0.1, 0.01, 0.005)
alpha_HW1 = st.sidebar.slider('correlation between haemoglobin and BMI', 0.0, 0.1, 0.01, 0.005)
alpha_AW1 = st.sidebar.slider('correlation between albumin and BMI', 0.0, 0.1, 0.01, 0.005)
alpha_IW1 = st.sidebar.slider('correlation between iron and BMI', 0.0, 0.1, 0.01, 0.005)
beta_HC1 = st.sidebar.slider('correlation between haemoglobin and CRP', 0.0, 0.1, 0.01, 0.005)
theta_HI1 = st.sidebar.slider('correlation between haemoglobin and iron', 0.0, 0.1, 0.01, 0.005)
delta1 = st.sidebar.slider('Delta', -1.0, 1.0, 0.3, 0.1)

noise_level = st.sidebar.slider('Noise Level', 0.0, 2.0, 1.0, 0.1)
add_noise = st.sidebar.checkbox('Add Noise', True)

# Initial conditions
st.sidebar.markdown("### values at birth (Intermediate Group)")
inter_mu_simu = st.sidebar.number_input('Mean of C CRP (Intermediate Group)', value=2.2)
mean_H_simu_inter = st.sidebar.number_input('Mean of haemoglobin (Intermediate Group)', value=10.6)
mean_W_simu_inter = st.sidebar.number_input('Mean of BMI (Intermediate Group)', value=14.0)
mean_A_simu_inter = st.sidebar.number_input('Mean of albumin (Intermediate Group)', value=3.8)
mean_I_simu_inter = st.sidebar.number_input('Mean of iron (Intermediate Group)', value=32.5)

st.sidebar.markdown("### values at birth (Severe Group)")
se_mu_simu = st.sidebar.number_input('Mean of C CRP (Severe Group)', value=3.9)
mean_H_simu_se = st.sidebar.number_input('Mean of haemoglobin (Severe Group)', value=8.4)
mean_W_simu_se = st.sidebar.number_input('Mean of BMI (Severe Group)', value=14.0)
mean_A_simu_se = st.sidebar.number_input('Mean of albumin (Severe Group)', value=2.8)
mean_I_simu_se = st.sidebar.number_input('Mean of iron (Severe Group)', value=16.4)

# Run simulation button
if st.sidebar.button('Run Simulation'):
    # Convert parameters to the appropriate format
    params1 = [r_C1, r_H1, r_W1, r_A1, r_I1,
               alpha_CW1, alpha_HW1, alpha_AW1, alpha_IW1,
               beta_HC1, theta_HI1, delta1]
    
    # Adjust noise level
    sigma_simu_prime = 1.3  # Adjusted based on your sigma_simu_prime
    noise_C_std = noise_level * sigma_simu_prime
    noise_H_std = noise_level * 2.5  # std_H
    noise_W_std = noise_level * 2.3  # std_W
    noise_A_std = noise_level * 0.9  # std_A
    noise_I_std = noise_level * 21   # std_I
    noise_std = [noise_C_std, noise_H_std, noise_W_std, noise_A_std, noise_I_std]

    # Generate initial conditions based on the input means
    num_patient = 6
    simu_init_inter = generate_initial_conditions(
        num_patient,
        inter_mu_simu,
        mean_H_simu_inter,
        mean_W_simu_inter,
        mean_A_simu_inter,
        mean_I_simu_inter
    )
    simu_init_se = generate_initial_conditions(
        num_patient,
        se_mu_simu,
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
    K_H = 14.0
    K_W = 20.0
    K_A = 5.0
    K_I = 110.0
    maxes = K_C, K_H, K_W, K_A, K_I

    # Run the simulation
    final_data = concatenate_data_diff_noise(
        simu_init_inter, simu_init_se, time_points_simu,
        maxes, params1, noise_std, noise=add_noise, type='hypo'
    )
    
    # Visualize the results
    st.markdown("## Simulation Results")
    plot_simulation(final_data)
else:
    st.markdown("## Awaiting Simulation")
    st.markdown("Adjust parameters and click **Run Simulation** to see results.")

