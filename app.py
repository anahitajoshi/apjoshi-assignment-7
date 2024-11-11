
from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

app = Flask(__name__)
app.secret_key = "b9f3e4b6c8a1d4e2f5c7a8b2d4e9c0a7" 


def generate_data(N, mu, beta0, beta1, sigma2, S):

    X = np.random.uniform(0, 1, N)  # Replace with code to generate random values for X

    Y = beta0 + beta1 * X + mu + np.random.normal(0, np.sqrt(sigma2), N)  # Replace with code to generate Y

    model = LinearRegression().fit(X.reshape(-1, 1), Y) 
    slope = model.coef_[0]
    intercept = model.intercept_  

    plot1_path = "static/plot1.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, alpha=0.5, color="#4169E1", edgecolors="blue", label="Data points")
    plt.plot(X, model.predict(X.reshape(-1, 1)), color="red", label=f"Fitted Line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"y = {slope:.2f}x + {intercept:.2f}")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(plot1_path)
    plt.close()

    slopes = []
    intercepts = []

    for _ in range(S):
        X_sim = np.random.uniform(0, 1, N) 
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), N)  

        sim_model = LinearRegression().fit(X_sim.reshape(-1, 1), Y_sim)  
        sim_slope = sim_model.coef_[0]  
        sim_intercept = sim_model.intercept_  

        slopes.append(sim_slope)
        intercepts.append(sim_intercept)

    plot2_path = "static/plot2.png"
    plt.figure(figsize=(10, 6))
    plt.hist(slopes, bins=20, alpha=0.6, color="#4169E1", label="Slopes")  
    plt.hist(intercepts, bins=20, alpha=0.6, color="#DAA520", label="Intercepts") 

    plt.axvline(slope, color='blue', linestyle='--', linewidth=1.5, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color='orange', linestyle='--', linewidth=1.5, label=f"Intercept: {intercept:.2f}")

    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Slopes and Intercepts")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot2_path)
    plt.close()

    slope_more_extreme = np.mean(np.abs(np.array(slopes) - beta1) >= np.abs(slope - beta1))  # Replace with code to calculate proportion of slopes more extreme than observed
    intercept_extreme = np.mean(np.abs(np.array(intercepts) - beta0) >= np.abs(intercept - beta0))  # Replace with code to calculate proportion of intercepts more extreme than observed

    return (
        X, Y, slope,intercept, plot1_path, plot2_path, slope_more_extreme, intercept_extreme, slopes,intercepts,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    return index()


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    if test_type == "greater":
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "less":
        p_value = np.mean(simulated_stats <= observed_stat)
    else: 
        compareVal = abs(observed_stat - hypothesized_value)
        p_value = np.mean(abs(simulated_stats - hypothesized_value) >= compareVal)

    if p_value <= 0.0001:
        fun_message = "Incredible! p ≤ 0.0001 – this is like finding a needle in a haystack!"
    else:
        fun_message = None


    plot3_path = "static/plot3.png"
    plt.figure(figsize=(10, 6))
    plt.hist(simulated_stats, bins=20, alpha=0.7, label='Simulated Statistics')
    plt.axvline(observed_stat, color='red', linestyle='--', label=f'Observed {parameter}: {observed_stat:.4f}')
    plt.axvline(hypothesized_value, color='blue', label=f'Hypothesized {parameter} (H₀): {hypothesized_value}')
    plt.xlabel(parameter.capitalize())
    plt.ylabel('Frequency')
    plt.title(f'Hypothesis Test for {parameter.capitalize()}')
    plt.legend()
    plt.savefig(plot3_path)
    plt.close()


    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
         p_value=p_value,
         fun_message=fun_message,
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    X = np.array(session.get("X"))
    Y = np.array(session.get("Y"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = slope
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = intercept
        true_param = beta0

    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)

    ci_lower = mean_estimate - stats.t.ppf(1 - (1 - confidence_level / 100) / 2, len(estimates) - 1) * std_estimate / np.sqrt(len(estimates))
    ci_upper = mean_estimate + stats.t.ppf(1 - (1 - confidence_level / 100) / 2, len(estimates) - 1) * std_estimate / np.sqrt(len(estimates))

    includes_true = ci_lower <= true_param <= ci_upper

    plot4_path = "static/plot4.png"
    plt.figure(figsize=(10, 6))
    plt.scatter(estimates, [0]*len(estimates), alpha=0.3, color='gray', label='Simulated Estimates')
    plt.scatter(mean_estimate, 0, color='blue' if includes_true else 'red', s=100, label='Mean Estimate')
    plt.hlines(0, ci_lower, ci_upper, color='blue' if includes_true else 'red', linewidth=2, label=f'{confidence_level}% Confidence Interval')
    plt.axvline(true_param, color='green', linestyle='--', label=f'True {parameter}')
    plt.xlabel(f'{parameter.capitalize()} Estimate')
    plt.title(f'{confidence_level}% Confidence Interval for {parameter.capitalize()} (Mean Estimate)')
    plt.yticks([])
    plt.legend()
    plt.savefig(plot4_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )


if __name__ == "__main__":
    app.run(debug=True)